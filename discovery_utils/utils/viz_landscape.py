"""Generate quick data landscape visuals using BERTopic and UMAP."""

import os
import re

from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type

import altair as alt
import nltk
import numpy as np
import openai
import pandas as pd

from bertopic import BERTopic
from bertopic.representation import OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

from discovery_utils import logging
from discovery_utils.getters.crunchbase import CrunchbaseGetter
from discovery_utils.getters.crunchbase import country_to_region
from discovery_utils.getters.gtr import GtrGetter


nltk.download("wordnet")

alt.data_transformers.disable_max_rows()

LLM_MODEL = "gpt-4o-mini"
MIN_CLUSTER_SIZE = 50

N_KEYWORD_CLUSTERS = 35

RANDOM_STATE = 42

CUSTOM_STOPWORDS = ["httpswwwukriorgapplyforfundinghowwefundstudentships", "abstract", "gtr"]
FULL_STOPWORDS = stopwords.words("english") + CUSTOM_STOPWORDS
LEMMATIZER = WordNetLemmatizer()


def simple_tokenizer(text: str) -> List[str]:
    """Split the text into words"""
    return text.split()


def preproc(text: str, list_of_stopwords: List[str] = FULL_STOPWORDS) -> str:
    """Preprocess text by removing non-alphabetic characters, lowercasing, lemmatising, and removing stopwords"""
    text = re.sub(r"[^a-zA-Z ]+", "", text).lower()
    text = text.split()
    text = [LEMMATIZER.lemmatize(t) for t in text]
    text = [t for t in text if t not in list_of_stopwords]
    return " ".join(text)


def concat_texts_in_cluster(documents: Iterator[str], cluster_labels: Iterator) -> Dict:
    """
    Create a large text string for each cluster, by joining up the text strings (documents) belonging to the same cluster

    Args:
        documents: A list of text strings
        cluster_labels: A list of cluster labels, indicating the membership of the text strings
    Returns:
        A dictionary where keys are cluster labels, and values are cluster text documents
    """

    assert len(documents) == len(cluster_labels)
    doc_type = type(documents[0])

    cluster_text_dict = defaultdict(doc_type)
    for i, doc in enumerate(documents):
        if doc_type is str:
            cluster_text_dict[cluster_labels[i]] += doc + " "
        elif doc_type is list:
            cluster_text_dict[cluster_labels[i]] += doc
    return cluster_text_dict


def generate_cluster_keywords(
    documents: Iterator[str],
    cluster_labels: Iterator[int],
    n: int = 10,
    tokenizer: Callable = simple_tokenizer,
    max_df: float = 0.90,
    min_df: float = 0.01,
    Vectorizer: Type[TfidfVectorizer] = TfidfVectorizer,
) -> Dict:
    """
    Generate keywords that characterise the cluster, using the specified Vectorizer

    Args:
        documents: List of (preprocessed) text documents
        cluster_labels: List of integer cluster labels
        n: Number of top keywords to return
        Vectorizer: Vectorizer object to use (eg, TfidfVectorizer, CountVectorizer)
        tokenizer: Function to use to tokenise the input documents; by default splits the document into words
    Returns:
        Dictionary that maps cluster integer labels to a list of keywords
    """

    # Define vectorizer
    vectorizer = Vectorizer(
        analyzer="word",
        tokenizer=tokenizer,
        preprocessor=lambda x: x,
        token_pattern=None,
        max_df=max_df,
        min_df=min_df,
        max_features=10000,
    )

    # Create cluster text documents
    cluster_documents = concat_texts_in_cluster(documents, cluster_labels)
    unique_cluster_labels = list(cluster_documents.keys())

    # Apply the vectorizer
    token_score_matrix = vectorizer.fit_transform(list(cluster_documents.values()))

    # Create a token lookup dictionary
    id_to_token = dict(zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys())))

    # For each cluster, check the top n tokens
    top_cluster_tokens = {}
    for i in range(token_score_matrix.shape[0]):
        # Get the cluster feature vector
        x = token_score_matrix[i, :].todense()
        # Find the indices of the top n tokens
        x = list(np.flip(np.argsort(np.array(x)))[0])[0:n]
        # Find the tokens corresponding to the top n indices
        top_cluster_tokens[unique_cluster_labels[i]] = [id_to_token[j] for j in x]

    return top_cluster_tokens


def generate_bertopic(
    vectors_df: pd.DataFrame,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    nr_topics: int = 10,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
    ngram_range: Tuple[int, int] = (1, 1),
    cluster_selection_method: str = "leaf",
) -> Tuple[pd.DataFrame, BERTopic]:
    """Generate quick BERTopic representations

    Args:
        vectors_df (pd.DataFrame): DataFrame with columns 'text' and 'vector'
        min_cluster_size (int, optional): Minimum cluster size. Defaults to MIN_CLUSTER_SIZE.
        nr_topics (int, optional): Constrained to number of topics to generate. Defaults to 10.
            You can set it to None to let the HDBSCAN determine the number of topics.
        random_state (int, optional): Random seed. Defaults to 42.
        verbose (bool, optional): Verbose mode. Defaults to False.
        ngram_range (Tuple[int, int], optional): N-gram range for the vectorizer. Defaults to (1, 1).
        cluster_selection_method (str, optional): Method to select clusters. Defaults to "leaf".

    Returns:
        pd.DataFrame: DataFrame with columns 'text', 'vector', 'topics', 'reduced_topics'
    """

    # Use OpenAI for cluster names
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    representation_model = OpenAI(client, model=LLM_MODEL, delay_in_seconds=0.5, chat=True)

    # Custom class to reduce frequent words in topic representations
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    # TO DO: do I still need this

    custom_hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        prediction_data=True,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean",
    )

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="euclidean", low_memory=False, random_state=random_state
    )

    topic_model = BERTopic(
        min_topic_size=min_cluster_size,
        n_gram_range=ngram_range,
        verbose=verbose,
        hdbscan_model=custom_hdbscan,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
    )

    # Fit the model using precomputed embeddings
    topics, probs = topic_model.fit_transform(
        vectors_df["text"],
        embeddings=np.array(vectors_df["vector"].to_list()),
    )
    vectors_df["topics"] = topics

    # Assign noise points to other topics
    try:
        reduced_topics = topic_model.reduce_outliers(
            vectors_df["text"].to_list(),
            topics=vectors_df["topics"].to_list(),
            strategy="embeddings",
            embeddings=np.array(vectors_df["vector"].to_list()),
        )
        vectors_df["reduced_topics"] = reduced_topics
        logging.info("Outliers were successfully reduced")
    except Exception as e:
        vectors_df["reduced_topics"] = topics
        logging.error(f"An error occurred while reducing outliers: {e}")
    vectors_df = vectors_df.assign(reduced_topics=lambda df: df.reduced_topics.astype(int))
    return vectors_df, topic_model


def generate_landscape_keywords(
    viz_df: pd.DataFrame,
    n_keyword_clusters: int = N_KEYWORD_CLUSTERS,
    random_state: int = RANDOM_STATE,
    x_col: str = "umap_x",
    y_col: str = "umap_y",
    text_col: str = "text",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate keywords for the landscape"""
    clusterer = KMeans(n_clusters=n_keyword_clusters, random_state=random_state)
    clusterer.fit(viz_df[[x_col, y_col]])
    soft_clusters = list(clusterer.labels_)

    title_texts = viz_df[text_col].apply(preproc)
    _cluster_texts = concat_texts_in_cluster(title_texts, soft_clusters)
    _cluster_keywords = generate_cluster_keywords(
        documents=list(_cluster_texts.values()),
        cluster_labels=list(_cluster_texts.keys()),
        n=2,
        max_df=0.90,
        min_df=0.01,
        Vectorizer=TfidfVectorizer,
    )
    viz_df["soft_cluster"] = soft_clusters
    viz_df["soft_cluster_"] = [str(x) for x in soft_clusters]
    viz_df["keyword_cluster"] = viz_df["soft_cluster"].apply(lambda x: ", ".join(_cluster_keywords[x]))

    centroids = (
        viz_df.groupby("soft_cluster")
        .agg(x_c=("umap_x", "mean"), y_c=("umap_y", "mean"))
        .reset_index()
        .assign(keywords=lambda x: x.soft_cluster.apply(lambda y: ", ".join(_cluster_keywords[y])))
    )

    return viz_df.drop(["soft_cluster", "soft_cluster_"], axis=1), centroids


def chart_keywords(centroids: pd.DataFrame) -> alt.Chart:
    """Generate a chart with keywords"""
    return (
        alt.Chart(centroids)
        .mark_text(
            fontSize=13.5,
            fontStyle="bold",
            opacity=0.8,
            stroke="white",
            strokeWidth=1,
            strokeOffset=0,
            strokeOpacity=0.4,
        )
        .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
    )


def generate_reduced_embeddings(vectors_df: pd.DataFrame, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Generate reduced embeddings for visualisation purposes"""
    return UMAP(
        n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=random_state
    ).fit_transform(np.array(vectors_df["vector"].to_list()))


def create_viz_dataframe(
    vectors_df: pd.DataFrame, reduced_embeddings: np.ndarray, topic_model: BERTopic
) -> pd.DataFrame:
    """Create a DataFrame for visualisation purposes

    Args:
        vectors_df (pd.DataFrame): DataFrame with columns 'text', 'vector'
        reduced_embeddings (np.ndarray): 2D array with reduced embeddings
        topic_model (BERTopic): BERTopic model

    Returns:
        Combined dataframe
    """
    return vectors_df.assign(
        umap_x=reduced_embeddings[:, 0],
        umap_y=reduced_embeddings[:, 1],
    ).merge(topic_model.get_topic_info(), left_on="reduced_topics", right_on="Topic", how="left")


def generate_landscape_viz_df(
    vectors_df: pd.DataFrame,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    nr_topics: int = 10,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
    n_keyword_clusters: int = N_KEYWORD_CLUSTERS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the data for landscape visualisation

    Args:
        vectors_df (pd.DataFrame): DataFrame with columns 'text' and 'vector'
        min_cluster_size (int, optional): Minimum cluster size. Defaults to MIN_CLUSTER_SIZE.
        nr_topics (int, optional): Number of topics to generate. Defaults to 10.
            Can set it to None to let the HDBSCAN determine the number of topics.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        Viz dataframe and centroids dataframe
    """
    _vectors_df, topic_model = generate_bertopic(
        vectors_df,
        min_cluster_size=min_cluster_size,
        nr_topics=nr_topics,
        random_state=random_state,
        verbose=verbose,
    )
    reduced_embeddings = generate_reduced_embeddings(_vectors_df, random_state)
    viz_df = create_viz_dataframe(_vectors_df, reduced_embeddings, topic_model)
    viz_df, centroids_df = generate_landscape_keywords(viz_df, n_keyword_clusters=n_keyword_clusters)
    return viz_df.drop(["vector", "Topic", "Representation"], axis=1), centroids_df


def scatter_keyword_chart(
    scatter_chart: alt.Chart,
    keyword_chart: alt.Chart,
    title: str = "",
    subtitle: List[str] = None,
) -> alt.Chart:
    """Combine scatter and keyword charts

    Args:
        scatter_chart (alt.Chart): Scatter chart
        keyword_chart (alt.Chart): Keyword chart
        title (str, optional): Title. Defaults to "".
        subtitle (List[str], optional): Subtitle (define multiple lines as strings in a list). Defaults to [""].
    """
    if subtitle is None:
        subtitle = [""]

    return (
        (scatter_chart + keyword_chart)
        .configure_axis(
            gridColor="white",
            domain=False,
        )
        .configure_view(strokeWidth=0, strokeOpacity=0)
        .properties(
            title={
                "anchor": "start",
                "text": title,
                "subtitle": subtitle,
                "subtitleFontSize": 14,
            },
        )
        .interactive()
    )


def crunchbase_dataframe(viz_df: pd.DataFrame, CB: CrunchbaseGetter) -> pd.DataFrame:
    """Create the final visualisation dataframe for Crunchbase data"""
    orgs_df = CB.organisations_enriched.query("id in @viz_df.id.to_list()")[
        ["id", "last_funding_on", "total_funding_gbp"]
    ]
    return (
        viz_df.merge(orgs_df, on="id", how="left")
        .assign(recent_funding=lambda x: x.last_funding_on > "2019")
        .astype({"recent_funding": str})
        .rename(columns={"name": "title", "text": "description", "Name": "category"})
        .fillna({"total_funding_gbp": 0})
        .assign(total_funding_gbp=lambda df: df.total_funding_gbp.apply(lambda x: round(x / 1e3, 3)))
        .assign(region=lambda x: x.country_code.apply(lambda y: country_to_region().get(y, "Rest of the World")))
    )


def chart_crunchbase_landscape(
    viz_df: pd.DataFrame,
    width: int = 900,
    height: int = 750,
    _opacity: float = 0.5,
) -> alt.Chart:
    """Generate the Crunchbase landscape visualisation"""

    # Dropdown menus
    name_dropdown = alt.binding_select(
        options=[None] + list(sorted(list(viz_df["category"].unique()))), name="Category:"
    )
    name_selection = alt.selection_point(fields=["category"], bind=name_dropdown, name="SelectName")

    recent_funding_dropdown = alt.binding_select(
        options=[None] + list(sorted(list(viz_df["recent_funding"].unique()))), name="Recent Funding:"
    )
    recent_funding_selection = alt.selection_point(
        fields=["recent_funding"], bind=recent_funding_dropdown, name="SelectFunding"
    )

    region_dropdown = alt.binding_select(
        options=[None] + list(sorted(list(viz_df["region"].unique()))), name="Region:"
    )
    region_selection = alt.selection_point(fields=["region"], bind=region_dropdown, name="SelectRegion")

    return (
        alt.Chart(viz_df, width=width, height=height)
        .mark_point(size=30, opacity=_opacity)
        .encode(
            x=alt.X("umap_x:Q", axis=None),
            y=alt.Y("umap_y:Q", axis=None),
            tooltip=[
                "title",
                "description",
                "country_code",
                "region",
                "category",
                "homepage_url",
                "last_funding_on",
                "total_funding_gbp",
            ],
            color=alt.Color("category", legend=alt.Legend(title="Category", labelLimit=300)),
            shape=alt.Shape("recent_funding", legend=alt.Legend(title="Recent funding (since 2020)")),
            opacity=alt.condition(
                recent_funding_selection & name_selection & region_selection, alt.value(_opacity), alt.value(0.0)
            ),
            href="homepage_url",
        )
        .add_params(
            recent_funding_selection,
            name_selection,
            region_selection,
        )
        .interactive()
    )


def generate_crunchbase_landscape(
    vectors_df: pd.DataFrame,
    CB: CrunchbaseGetter,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    nr_topics: int = 10,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
    n_keyword_clusters: int = N_KEYWORD_CLUSTERS,
) -> Tuple[alt.Chart, pd.DataFrame]:
    """Generate the Crunchbase landscape visualisation"""
    viz_df, centroids_df = generate_landscape_viz_df(
        vectors_df,
        min_cluster_size=min_cluster_size,
        nr_topics=nr_topics,
        random_state=random_state,
        verbose=verbose,
        n_keyword_clusters=n_keyword_clusters,
    )
    cb_viz_df = crunchbase_dataframe(viz_df, CB)
    fig = scatter_keyword_chart(
        scatter_chart=chart_crunchbase_landscape(cb_viz_df),
        keyword_chart=chart_keywords(centroids_df),
    )
    return fig, cb_viz_df


def gtr_dataframe(viz_df: pd.DataFrame, GTR: GtrGetter) -> pd.DataFrame:
    """Create the final visualisation dataframe for Crunchbase data"""
    projects_df = GTR.projects_enriched.query("id in @viz_df.id.to_list()")[["id", "leadFunder", "amount"]]
    return (
        viz_df.merge(projects_df, on="id", how="left")
        .assign(recent_project=lambda x: x.start > "2019")
        .astype({"recent_project": str})
        .rename(columns={"text": "description", "Name": "category"})
        .fillna({"amount": 0})
        .assign(amount=lambda df: df.amount.apply(lambda x: round(x / 1e3, 3)))
    )


def chart_gtr_landscape(
    viz_df: pd.DataFrame,
    width: int = 900,
    height: int = 750,
    _opacity: float = 0.5,
) -> alt.Chart:
    """Generate the Crunchbase landscape visualisation"""

    # Dropdown menus
    name_dropdown = alt.binding_select(
        options=[None] + list(sorted(list(viz_df["category"].unique()))), name="Category:"
    )
    name_selection = alt.selection_point(fields=["category"], bind=name_dropdown, name="SelectName")

    recent_funding_dropdown = alt.binding_select(
        options=[None] + list(sorted(list(viz_df["recent_project"].unique()))), name="Recent Project:"
    )
    recent_funding_selection = alt.selection_point(
        fields=["recent_project"], bind=recent_funding_dropdown, name="SelectFunding"
    )

    lead_funder_dropdown = alt.binding_select(
        options=[None] + list(sorted(list(viz_df["leadFunder"].unique()))), name="Lead funder:"
    )
    lead_funder_selection = alt.selection_point(fields=["leadFunder"], bind=lead_funder_dropdown, name="Lead funder")

    return (
        alt.Chart(viz_df, width=width, height=height)
        .mark_point(size=30, opacity=_opacity)
        .encode(
            x=alt.X("umap_x:Q", axis=None),
            y=alt.Y("umap_y:Q", axis=None),
            tooltip=["title", "description", "category", "leadFunder", "start", "end", "amount"],
            color=alt.Color("category", legend=alt.Legend(title="Category", labelLimit=300)),
            shape=alt.Shape("recent_project", legend=alt.Legend(title="Recent project (started since 2020)")),
            opacity=alt.condition(
                recent_funding_selection & name_selection & lead_funder_selection, alt.value(_opacity), alt.value(0.0)
            ),
            href="url",
        )
        .add_params(
            recent_funding_selection,
            name_selection,
            lead_funder_selection,
        )
        .interactive()
    )


def generate_gtr_landscape(
    vectors_df: pd.DataFrame,
    GTR: GtrGetter,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    nr_topics: int = 10,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
    n_keyword_clusters: int = N_KEYWORD_CLUSTERS,
) -> Tuple[alt.Chart, pd.DataFrame]:
    """Generate the Crunchbase landscape visualisation"""
    viz_df, centroids_df = generate_landscape_viz_df(
        vectors_df,
        min_cluster_size=min_cluster_size,
        nr_topics=nr_topics,
        random_state=random_state,
        verbose=verbose,
        n_keyword_clusters=n_keyword_clusters,
    )
    gtr_viz_df = gtr_dataframe(viz_df, GTR)
    fig = scatter_keyword_chart(
        scatter_chart=chart_gtr_landscape(gtr_viz_df),
        keyword_chart=chart_keywords(centroids_df),
    )
    return fig, gtr_viz_df
