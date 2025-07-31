"""
Overton API client for policy document retrieval and analysis.

This module provides a interface to the Overton API for searching
and analysing policy documents.

Example:
    >>> from discovery_utils.getters.overton import OvertonGetter
    >>> overton = OvertonGetter()
    >>> docs = overton.search_documents("climate change", max_results=100)
    >>> print(f"Found {len(docs)} documents")
"""

import logging
import os
import threading
import time

from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import requests

from dotenv import load_dotenv

from discovery_utils import logging as du_logging
from discovery_utils.utils import s3


# Load environment variables
load_dotenv()


class OvertonAPIError(Exception):
    """Base exception for Overton API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class OvertonRateLimitError(OvertonAPIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, message: str = "API rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class OvertonAuthError(OvertonAPIError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed - invalid API key"):
        super().__init__(message, status_code=401)


class OvertonValidationError(OvertonAPIError):
    """Raised when parameter validation fails."""

    def __init__(self, message: str, parameter: Optional[str] = None):
        super().__init__(message)
        self.parameter = parameter


class OvertonConnectionError(OvertonAPIError):
    """Raised when connection to API fails."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class OvertonRateLimiter:
    """Thread-safe rate limiter for Overton API compliance.

    Ensures strict compliance with Overton's 1 request/second rate limit
    while providing thread-safe operation for concurrent usage.
    """

    def __init__(self, requests_per_second: float = 1.0):
        """Initialise rate limiter.

        Args:
            requests_per_second (float): Maximum requests per second.
                Defaults to 1.0 for Overton API compliance.
        """
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")

        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Ensure rate limit compliance before making request.

        This method blocks if necessary to maintain the configured rate limit.
        Thread-safe for concurrent usage.
        """
        with self._lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


class OvertonGetter:
    """Main interface for Overton API access.

    Provides access to Overton's document database with functionality
    including rate limiting, error handling, pagination, and standardised data
    output formats.

    Attributes:
        BASE_URL (str): Base URL for Overton API endpoints
        REGION_MAPPINGS (Dict[str, str]): Special region filters mapping
    """

    BASE_URL = "https://app.overton.io"

    # Special region mappings for frontend labels
    REGION_MAPPINGS = {
        "All but UK": "_:uxf",
        "OECD members": "OECD members",
        "Non-OECD members": "Non-OECD members",
        "G20": "G20",
        "G7": "G7",
        "North America": "North America",
        "South and Central America": "South and Central America",
        "Europe": "Europe",
        "Nordics": "Nordics",
        "APAC": "APAC",
        "Africa": "Africa",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        cache_enabled: bool = False,
        cache_ttl: int = 3600,
    ):
        """Initialise Overton API client.

        Args:
            api_key (str, optional): Overton API key. If None, will attempt to
                read from OVERTON_API_KEY environment variable.
            rate_limit (float, optional): Requests per second limit. Defaults to 1.0
                for strict Overton compliance.
            timeout (int, optional): Request timeout in seconds. Defaults to 30.
            max_retries (int, optional): Maximum retry attempts for failed requests.
                Defaults to 3.
            backoff_factor (float, optional): Exponential backoff factor for retries.
                Defaults to 1.0.
            cache_enabled (bool, optional): Enable response caching. Defaults to False.
            cache_ttl (int, optional): Cache time-to-live in seconds. Defaults to 3600.

        Raises:
            OvertonAuthError: If no API key is provided or found in environment.
            OvertonValidationError: If parameters are invalid.
        """
        # API key with environment fallback
        self.api_key = api_key or os.getenv("OVERTON_API_KEY")
        if not self.api_key:
            raise OvertonAuthError(
                "API key required. Set OVERTON_API_KEY environment variable " "or pass api_key parameter."
            )

        # Validate parameters
        if rate_limit <= 0:
            raise OvertonValidationError("rate_limit must be positive", "rate_limit")
        if timeout <= 0:
            raise OvertonValidationError("timeout must be positive", "timeout")
        if max_retries < 0:
            raise OvertonValidationError("max_retries must be non-negative", "max_retries")
        if backoff_factor <= 0:
            raise OvertonValidationError("backoff_factor must be positive", "backoff_factor")

        # Rate limiting setup
        self.rate_limiter = OvertonRateLimiter(rate_limit)

        # HTTP session configuration
        self.session = requests.Session()
        self.session.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Set user agent for API identification
        self.session.headers.update({"User-Agent": "discovery-utils-overton-getter/1.0"})

        # Logging setup
        if du_logging:
            self.logger = du_logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        # Caching setup (optional)
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._cache = {} if cache_enabled else None

        # Property caching for common data
        self._recent_documents = None
        self._facets_cache = None
        self._facets_cache_time = None

        self.logger.info(f"OvertonGetter initialized with rate_limit={rate_limit}/s")

    def _make_request(
        self, endpoint: str, params: Dict[str, Any], method: str = "GET", validate_response: bool = True
    ) -> Dict[str, Any]:
        """Make rate-limited API request with retry logic.

        Args:
            endpoint (str): API endpoint (e.g., "documents.php")
            params (Dict[str, Any]): Request parameters
            method (str, optional): HTTP method. Defaults to "GET".
            validate_response (bool, optional): Whether to validate response structure.
                Defaults to True.

        Returns:
            Dict[str, Any]: JSON response from API

        Raises:
            OvertonAPIError: If request fails after all retries
            OvertonRateLimitError: If rate limit exceeded permanently
            OvertonAuthError: If authentication fails
            OvertonConnectionError: If connection fails
        """
        # Check cache first
        cache_key = self._generate_cache_key(endpoint, params)
        cached_response = self._get_cached_response(cache_key)
        if cached_response is not None:
            return cached_response

        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}/{endpoint}"
        request_params = {**params, "api_key": self.api_key, "format": "json"}

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Make request based on method
                if method.upper() == "GET":
                    response = self.session.get(url, params=request_params)
                elif method.upper() == "POST":
                    response = self.session.post(url, data=request_params)
                else:
                    raise OvertonValidationError(f"Unsupported HTTP method: {method}")

                # Handle different response codes
                if response.status_code == 200:
                    try:
                        json_response = response.json()

                        # Validate response structure if requested
                        if validate_response and not self._validate_response_structure(json_response):
                            raise OvertonAPIError("Invalid response structure")

                        # Cache successful response
                        self._cache_response(cache_key, json_response)

                        self.logger.debug(f"Successful request to {endpoint}")
                        return json_response

                    except ValueError as e:
                        raise OvertonAPIError(f"Invalid JSON response: {e}")

                elif response.status_code == 429:
                    # Rate limit exceeded
                    retry_after = int(response.headers.get("Retry-After", 60))
                    wait_time = min(retry_after, (2**attempt) * self.backoff_factor)

                    if attempt == self.max_retries:
                        raise OvertonRateLimitError(
                            f"Rate limit exceeded after {self.max_retries} attempts", retry_after=retry_after
                        )

                    self.logger.warning(f"Rate limit hit (attempt {attempt + 1}), waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 401:
                    raise OvertonAuthError("Invalid API key or authentication failed")

                elif response.status_code == 403:
                    raise OvertonAuthError("Access forbidden - check API key permissions")

                elif response.status_code == 404:
                    raise OvertonAPIError(f"Endpoint not found: {endpoint}")

                else:
                    # Other HTTP errors
                    error_msg = f"HTTP {response.status_code}: {response.reason}"
                    if attempt == self.max_retries:
                        raise OvertonAPIError(error_msg, status_code=response.status_code)

                    self.logger.warning(f"HTTP error {response.status_code} (attempt {attempt + 1}), retrying")

            except requests.ConnectionError as e:
                last_exception = OvertonConnectionError(f"Connection failed: {e}", e)
                if attempt == self.max_retries:
                    raise last_exception

            except requests.Timeout as e:
                last_exception = OvertonConnectionError(f"Request timeout: {e}", e)
                if attempt == self.max_retries:
                    raise last_exception

            except requests.RequestException as e:
                last_exception = OvertonAPIError(f"Request failed: {e}")
                if attempt == self.max_retries:
                    raise last_exception

            # Wait before retry
            if attempt < self.max_retries:
                wait_time = self.backoff_factor * (2**attempt)
                self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s")
                time.sleep(wait_time)

        # Should not reach here, but just in case
        raise last_exception or OvertonAPIError("Request failed after all retries")

    def _validate_response_structure(self, response: Dict[str, Any]) -> bool:
        """Validate basic response structure.

        Args:
            response (Dict[str, Any]): API response

        Returns:
            bool: True if response structure is valid
        """
        # Basic structure validation
        if not isinstance(response, dict):
            return False

        # For search responses, expect 'results' field
        if "results" in response:
            return isinstance(response["results"], list)

        # For other responses, just check it's a dict
        return True

    def _build_search_params(
        self,
        query: Optional[str] = None,
        semantic_search: bool = False,
        source_country: Optional[str] = None,
        source_region: Optional[str] = None,
        source_type: Optional[str] = None,
        published_after: Optional[Union[date, str]] = None,
        published_before: Optional[Union[date, str]] = None,
        topics: Optional[str] = None,
        classifications: Optional[str] = None,
        plain_dois_cited: Optional[str] = None,
        min_similarity: float = 0.3,
        sort: str = "relevance",
        page: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build comprehensive search parameters for API request.

        Args:
            query (str, optional): Search query string
            semantic_search (bool, optional): Use semantic search. Defaults to False.
            source_country (str, optional): Filter by source country
            source_region (str, optional): Filter by source region
            source_type (str, optional): Filter by source type
            published_after (Union[date, str], optional): Start date filter
            published_before (Union[date, str], optional): End date filter
            topics (str, optional): Topic filter
            classifications (str, optional): Classification filter
            plain_dois_cited (str, optional): DOI citation filter
            min_similarity (float, optional): Minimum similarity for semantic search
            sort (str, optional): Sort order. Defaults to "relevance".
            page (int, optional): Page number. Defaults to 1.
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Processed parameters for API request
        """
        params = {}

        # Query type selection
        if query:
            params["squery" if semantic_search else "query"] = query

        # Semantic search parameters
        if semantic_search and min_similarity != 0.3:
            params["min_similarity"] = min_similarity

        # Country/region handling with special mappings
        if source_country:
            if source_country == "All":
                # Do not set any filter for 'All'
                pass
            elif source_country in self.REGION_MAPPINGS:
                params["source_region"] = self.REGION_MAPPINGS[source_country]
            else:
                params["source_country"] = source_country

        if source_region:
            params["source_region"] = source_region

        # Source type handling
        if source_type and source_type.strip() and source_type.lower() != "all":
            params["source_type"] = source_type

        # Date handling
        for date_field, date_value in [("published_after", published_after), ("published_before", published_before)]:
            if date_value:
                if isinstance(date_value, date):
                    params[date_field] = date_value.isoformat()
                else:
                    params[date_field] = str(date_value)

        # Topic and classification filters
        if topics:
            params["topics"] = topics
        if classifications:
            params["classifications"] = classifications
        if plain_dois_cited:
            params["plain_dois_cited"] = plain_dois_cited

        # Sorting and pagination
        params["sort"] = sort
        params["page"] = page

        # Add any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value

        return params

    def _validate_api_key(self) -> bool:
        """Validate API key by making a test request.

        Returns:
            bool: True if API key is valid, False otherwise.
        """
        try:
            response = self._make_request("documents.php", {"query": "test", "page": 1}, validate_response=False)
            return response is not None
        except OvertonAuthError:
            return False
        except Exception:
            # Other errors don't necessarily mean invalid API key
            return True

    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request parameters.

        Args:
            endpoint (str): API endpoint
            params (Dict[str, Any]): Request parameters

        Returns:
            str: Cache key
        """
        # Remove API key from cache key for security
        cache_params = {k: v for k, v in params.items() if k != "api_key"}

        # Sort parameters for consistent cache keys
        sorted_params = sorted(cache_params.items())
        param_str = "&".join(f"{k}={v}" for k, v in sorted_params)

        return f"{endpoint}?{param_str}"

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired.

        Args:
            cache_key (str): Cache key

        Returns:
            Optional[Dict[str, Any]]: Cached response or None
        """
        if not self.cache_enabled or not self._cache:
            return None

        if cache_key in self._cache:
            response, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.logger.debug(f"Cache hit for {cache_key}")
                return response
            else:
                # Remove expired entry
                del self._cache[cache_key]
                self.logger.debug(f"Cache expired for {cache_key}")

        return None

    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Cache response with timestamp.

        Args:
            cache_key (str): Cache key
            response (Dict[str, Any]): Response to cache
        """
        if self.cache_enabled and self._cache is not None:
            self._cache[cache_key] = (response, time.time())
            self.logger.debug(f"Cached response for {cache_key}")

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        if self._cache:
            self._cache.clear()
            self.logger.info("Response cache cleared")

        # Clear property caches
        self._recent_documents = None
        self._facets_cache = None
        self._facets_cache_time = None
        self.logger.info("Property caches cleared")

    @property
    def recent_documents(self) -> pd.DataFrame:
        """Get recently published policy documents.

        Returns documents published in the last 30 days, sorted by publication date.
        Results are cached for performance.

        Returns:
            pd.DataFrame: Recent policy documents
        """
        if self._recent_documents is None:
            thirty_days_ago = date.today() - timedelta(days=30)
            self._recent_documents = self.search_documents(
                published_after=thirty_days_ago, sort="date", max_results=200
            )
            self.logger.info(f"Loaded {len(self._recent_documents)} recent documents")

        return self._recent_documents

    @property
    def facets(self) -> Dict[str, List[Dict]]:
        """Get current facet information.

        Returns available facets for filtering searches. Results are cached
        for performance with 1-hour TTL.

        Returns:
            Dict[str, List[Dict]]: Facet information by field
        """
        current_time = time.time()

        # Check if cache is still valid (1 hour TTL)
        if (
            self._facets_cache is not None
            and self._facets_cache_time is not None
            and current_time - self._facets_cache_time < 3600
        ):
            return self._facets_cache

        # Refresh facets
        self._facets_cache = self.get_facets()
        self._facets_cache_time = current_time

        return self._facets_cache

    def __repr__(self) -> str:
        """String representation of OvertonGetter instance."""
        return (
            f"OvertonGetter(rate_limit={1.0/self.rate_limiter.min_interval}/s, " f"cache_enabled={self.cache_enabled})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Overton API Client (Rate: {1.0/self.rate_limiter.min_interval}/s)"

    def search_documents(
        self,
        query: Optional[str] = None,
        semantic_search: bool = False,
        max_results: int = 1000,
        source_country: Optional[str] = None,
        source_type: Optional[str] = None,
        published_after: Optional[Union[date, str]] = None,
        published_before: Optional[Union[date, str]] = None,
        topics: Optional[str] = None,
        classifications: Optional[str] = None,
        sort: str = "relevance",
        page: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Search policy documents using Overton API.

        Provides comprehensive search capabilities across Overton's policy
        document database with support for text search, semantic search,
        and advanced filtering options.

        Args:
            query (str, optional): Search query string. If None, returns
                recent documents based on other filters.
            semantic_search (bool, optional): Use semantic search instead
                of keyword search. Defaults to False.
            max_results (int, optional): Maximum number of results to return.
                Defaults to 1000. API enforces pagination limits.
            source_country (str, optional): Filter by country/region. Supports
                special values like "UK", "All but UK", "OECD members", etc.
            source_type (str, optional): Filter by source type (e.g., "government").
            published_after (Union[date, str], optional): Start date for document publication.
            published_before (Union[date, str], optional): End date for document publication.
            topics (str, optional): Topic filter string.
            classifications (str, optional): Classification filter string.
            sort (str, optional): Sort order. Options: "relevance", "date". Defaults to "relevance".
            page (int, optional): Starting page number. Defaults to 1.
            **kwargs: Additional search parameters for the Overton API.

        Returns:
            pd.DataFrame: Standardised DataFrame with columns:
                - id: Policy document ID
                - title: Document title
                - abstract: Document abstract/snippet
                - content: Truncated content for analysis
                - authors: List of authors
                - publication_year: Year of publication
                - venue: Publishing organisation
                - topics: List of document topics
                - source_country: Source country
                - source_type: Source type
                - And additional metadata fields

        Raises:
            OvertonAPIError: If API request fails
            OvertonRateLimitError: If rate limit exceeded
            OvertonAuthError: If authentication fails
            OvertonValidationError: If parameters are invalid

        Example:
            >>> overton = OvertonGetter(api_key="your_key")
            >>> docs = overton.search_documents(
            ...     "climate change",
            ...     source_country="UK",
            ...     published_after=date(2023, 1, 1),
            ...     max_results=100
            ... )
            >>> print(f"Found {len(docs)} documents")
        """
        # Parameter validation
        if max_results <= 0:
            raise OvertonValidationError("max_results must be positive", "max_results")
        if page < 1:
            raise OvertonValidationError("page must be >= 1", "page")
        if sort not in ["relevance", "date"]:
            raise OvertonValidationError("sort must be 'relevance' or 'date'", "sort")

        self.logger.info(f"Searching documents: query='{query}', max_results={max_results}")

        # Build search parameters
        params = self._build_search_params(
            query=query,
            semantic_search=semantic_search,
            source_country=source_country,
            source_type=source_type,
            published_after=published_after,
            published_before=published_before,
            topics=topics,
            classifications=classifications,
            sort=sort,
            page=page,
            **kwargs,
        )

        # Handle single page vs multi-page requests
        if max_results <= 200:  # Single page request
            response = self._make_request("documents.php", params)
            documents = response.get("results", [])
            limited_documents = documents[:max_results]

            self.logger.info(f"Retrieved {len(limited_documents)} documents (single page)")
            return self._process_documents(limited_documents)

        # Multi-page request with pagination
        return self._paginate_search(params, max_results)

    def _paginate_search(self, base_params: Dict[str, Any], max_results: int) -> pd.DataFrame:
        """Efficiently paginate through large result sets.

        Args:
            base_params (Dict[str, Any]): Base search parameters
            max_results (int): Maximum number of results to collect

        Returns:
            pd.DataFrame: All collected documents processed into DataFrame
        """
        all_documents = []
        page = base_params.get("page", 1)

        self.logger.info(f"Starting paginated search for {max_results} documents")

        while len(all_documents) < max_results:
            # Update page parameter
            params = {**base_params, "page": page}

            try:
                response = self._make_request("documents.php", params)
                documents = response.get("results", [])

                if not documents:
                    self.logger.info(f"No more documents found at page {page}")
                    break

                all_documents.extend(documents)
                self.logger.debug(
                    f"Page {page}: collected {len(documents)} documents " f"(total: {len(all_documents)})"
                )

                # Check for next page URL in response
                query_info = response.get("query", {})
                if not query_info.get("next_page_url"):
                    self.logger.info(f"No next page URL found, stopping pagination")
                    break

                page += 1

                # Break if we've collected enough documents
                if len(all_documents) >= max_results:
                    break

            except OvertonAPIError as e:
                self.logger.error(f"Error during pagination at page {page}: {e}")
                break

        # Limit to requested number of results
        final_documents = all_documents[:max_results]
        self.logger.info(f"Pagination complete: collected {len(final_documents)} documents")

        return self._process_documents(final_documents)

    def _process_documents(self, documents: List[Dict]) -> pd.DataFrame:
        """Convert API response documents to standardised DataFrame format.

        Args:
            documents (List[Dict]): Raw document data from API

        Returns:
            pd.DataFrame: Processed and normalised document data
        """
        if not documents:
            self.logger.warning("No documents to process")
            return pd.DataFrame()

        processed = []

        for doc in documents:
            try:
                # Author normalisation
                authors = doc.get("authors", [])
                if isinstance(authors, str):
                    authors = [authors] if authors else []
                elif not isinstance(authors, list):
                    authors = []

                # Topic normalisation
                topics = doc.get("topics", [])
                if isinstance(topics, str):
                    topics = [topics] if topics else []
                elif not isinstance(topics, list):
                    topics = []

                # Content aggregation from multiple fields
                content_fields = ["snippet", "llm_document_description", "abstract"]
                content_parts = [doc.get(field, "").strip() for field in content_fields if doc.get(field)]
                full_content = " ".join(content_parts).strip()

                # Extract publication year safely
                published_on = doc.get("published_on", "")
                publication_year = ""
                if published_on:
                    try:
                        publication_year = published_on.split("-")[0]
                    except (IndexError, AttributeError):
                        pass

                # Extract source information safely
                source_info = doc.get("source", {})
                if not isinstance(source_info, dict):
                    source_info = {}

                result = {
                    "id": doc.get("policy_document_id", ""),
                    "title": doc.get("title", ""),
                    "abstract": full_content or "No abstract available",
                    "content": self._truncate_content(full_content, 1000),
                    "authors": authors,
                    "publication_year": publication_year,
                    "venue": source_info.get("title", ""),
                    "doi": doc.get("document_url", ""),
                    "citation_count": self._safe_int_conversion(doc.get("citation_count", 0)),
                    "topics": topics,
                    "source_country": source_info.get("country", ""),
                    "source_type": source_info.get("type", ""),
                    "published_on": published_on,
                    "overton_url": doc.get("overton_url", ""),
                    "pdf_url": doc.get("pdf_url", ""),
                    "similarity_score": doc.get("similarity_score"),
                }
                processed.append(result)

            except Exception as e:
                self.logger.warning(f"Error processing document {doc.get('policy_document_id', 'unknown')}: {e}")
                continue

        if not processed:
            self.logger.warning("No documents successfully processed")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(processed)

        # Data cleaning and normalisation
        df["abstract"] = df["abstract"].fillna("No abstract available")
        df["content"] = df["content"].fillna("No content available")
        df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0)

        # Ensure authors and topics are lists
        df["authors"] = df["authors"].apply(lambda x: x if isinstance(x, list) else [])
        df["topics"] = df["topics"].apply(lambda x: x if isinstance(x, list) else [])

        self.logger.debug(f"Processed {len(df)} documents successfully")
        return df

    def _truncate_content(self, content: str, max_length: int = 1000) -> str:
        """Truncate content to specified length with ellipsis.

        Args:
            content (str): Content to truncate
            max_length (int): Maximum length. Defaults to 1000.

        Returns:
            str: Truncated content
        """
        if not content:
            return "No content available"

        if len(content) <= max_length:
            return content

        # Truncate and add ellipsis
        return content[:max_length].rstrip() + "..."

    def _safe_int_conversion(self, value: Any) -> int:
        """Safely convert value to integer.

        Args:
            value (Any): Value to convert

        Returns:
            int: Converted integer or 0 if conversion fails
        """
        try:
            if value is None:
                return 0
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def get_facets(self, query: Optional[str] = None, clear_cache: bool = False) -> Dict[str, List[Dict]]:
        """Get facet information for filtering searches.

        Retrieves available facets and their counts, which can be used
        to understand the data distribution and build effective filters.

        Args:
            query (str, optional): Apply query to get facets for specific search.
                If None, returns general facets.
            clear_cache (bool, optional): Force refresh of cached facets.
                Defaults to False.

        Returns:
            Dict[str, List[Dict]]: Facet information by field name.
                Each facet contains 'key' and 'doc_count' fields.

        Example:
            >>> overton = OvertonGetter()
            >>> facets = overton.get_facets("climate change")
            >>> countries = facets.get("policy_source_country", [])
            >>> print(f"Top country: {countries[0]['key']} ({countries[0]['doc_count']} docs)")
        """
        # Check if we should use cached facets
        if self._facets_cache and not clear_cache and not query:
            return self._facets_cache

        params = {}
        if query:
            params["query"] = query

        try:
            response = self._make_request("documents.php", params)
            facets = response.get("facets", {})

            # Cache general facets (without query)
            if not query:
                self._facets_cache = facets
                self._facets_cache_time = time.time()

            self.logger.debug(f"Retrieved facets with {len(facets)} fields")
            return facets

        except OvertonAPIError as e:
            self.logger.error(f"Failed to retrieve facets: {e}")
            return {}

    def generate_id_set(self, identifiers: List[str], identifier_type: str = "dois") -> str:
        """Generate ID set for bulk operations with document identifiers.

        Creates a set of identifiers that can be used in subsequent searches
        to find documents that cite or reference specific papers.

        Args:
            identifiers (List[str]): List of identifiers (DOIs, ISBNs, etc.)
            identifier_type (str, optional): Type of identifiers.
                Options: "dois", "isbns". Defaults to "dois".

        Returns:
            str: Set ID that can be used in search operations

        Raises:
            OvertonValidationError: If identifiers list is empty or invalid
            OvertonAPIError: If set generation fails

        Example:
            >>> overton = OvertonGetter()
            >>> dois = ["10.1038/nature12345", "10.1126/science.abc123"]
            >>> set_id = overton.generate_id_set(dois)
            >>> citing_docs = overton.search_with_id_set(set_id)
        """
        if not identifiers:
            raise OvertonValidationError("identifiers list cannot be empty", "identifiers")

        if identifier_type not in ["dois", "isbns"]:
            raise OvertonValidationError("identifier_type must be 'dois' or 'isbns'", "identifier_type")

        # Format identifiers for POST request
        ids_string = "\n".join(str(id_).strip() for id_ in identifiers if id_)

        if not ids_string:
            raise OvertonValidationError("No valid identifiers provided", "identifiers")

        params = {identifier_type: ids_string}

        try:
            response = self._make_request("generate_id_set.php", params, method="POST")

            if "set" not in response:
                raise OvertonAPIError("Failed to generate ID set - no set ID returned")

            set_id = response["set"]
            self.logger.info(f"Generated ID set {set_id} with {len(identifiers)} {identifier_type}")
            return set_id

        except OvertonAPIError as e:
            self.logger.error(f"Failed to generate ID set: {e}")
            raise

    def search_with_id_set(
        self, set_id: str, field: str = "plain_dois_cited", max_results: int = 1000, **kwargs
    ) -> pd.DataFrame:
        """Search using a generated ID set.

        Finds documents that cite or reference the papers in the specified ID set.

        Args:
            set_id (str): ID set generated by generate_id_set()
            field (str, optional): Field to apply the set to.
                Options: "plain_dois_cited", "plain_isbns_cited".
                Defaults to "plain_dois_cited".
            max_results (int, optional): Maximum results to return. Defaults to 1000.
            **kwargs: Additional search parameters

        Returns:
            pd.DataFrame: Documents that cite the papers in the ID set

        Example:
            >>> set_id = overton.generate_id_set(dois)
            >>> citing_docs = overton.search_with_id_set(set_id, max_results=500)
        """
        if not set_id:
            raise OvertonValidationError("set_id cannot be empty", "set_id")

        search_params = {field: set_id, "max_results": max_results}
        search_params.update(kwargs)

        self.logger.info(f"Searching with ID set {set_id} in field {field}")
        return self.search_documents(**search_params)

    def format_for_screening(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Format policy documents for LLM screening and analysis.

        Converts a DataFrame of documents into a format suitable for
        automated screening, content analysis, or LLM processing.

        Args:
            df (pd.DataFrame): DataFrame from search_documents()

        Returns:
            Dict[str, Dict[str, str]]: Dictionary with document IDs as keys
                and title/content as values for each document

        Example:
            >>> docs = overton.search_documents("climate policy")
            >>> screening_data = overton.format_for_screening(docs)
            >>> for doc_id, content in screening_data.items():
            ...     print(f"Title: {content['title']}")
        """
        if df.empty:
            return {}

        # Ensure required columns exist
        required_cols = ["id", "title", "content"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise OvertonValidationError(f"DataFrame missing required columns: {missing_cols}")

        # Create screening format
        screening_dict = df.set_index("id")[["title", "content"]].to_dict("index")

        self.logger.debug(f"Formatted {len(screening_dict)} documents for screening")
        return screening_dict
