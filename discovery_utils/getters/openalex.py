import os

import pandas as pd


USER_EMAIL = os.environ["USER_EMAIL"]


def get_openalex_works(
    research_question: str,
    user: str = USER_EMAIL,
    min_cites: str = ">4",
    n_works: int = 10000,
) -> pd.DataFrame:
    """Get works from OpenAlex API."""
    import pyalex

    # set pyalex config email
    pyalex.config["email"] = user

    query = pyalex.Works().search(research_question).filter(cited_by_count=min_cites)

    results = []
    for page in query.paginate(per_page=200, n_max=n_works):
        results.extend(page)

    for page in results:
        page["abstract"] = page["abstract"]

    df = pd.DataFrame(results).dropna(subset=["title", "abstract"]).drop(columns=["abstract_inverted_index"])

    return df
