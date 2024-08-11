# Notebook to implement and evaluate a pipeline that uses GPT-3.5 for multi-document summarisation.

This pipeline includes two steps.

    Step 1: The attend_prompt: retrieves relevant information from every document in your text corpus.
    Step 2: The summary_prompt: summarises the information retrieved from the attend_prompt.

## Attend_prompt evaluation
The attend_prompt is evaluated using the RAGAS metrics: faithfulness, answer relevance and context precision. For more information on RAGAS and the metrics they offer refer to: https://docs.ragas.io/en/latest/concepts/metrics/index.html

To effectively apply these metrics you need to transfrom the information used for implementing the attend_prompt into a dataset. This datasets includes the following information:

    1. Question: The questions asked within the attend_prompt.
    2. Contexts: The context the LLM used to answer the questions within the attend_prompt
    3. Answer: The LLM-generated answer to the questions asked within the attend_prompt.
    4. Ground_truths: The actual answer for every question asked within the attend_prompt.

It is recommended to evaluate the attend_prompt on a random sample of documents within your dataset.
Note: If you cannot get access to the ground_truth you can still construct the dataset, but without the ground_truth.

The attend_prompt in this notebook is evaluated by integrating RAGAS with Langfuse, enabling you to track RAGAS scores for various metrics directly on Langfuse. This setup facilitates effective comparison of RAGAS scores when adjusting parameter settings or prompts within your attend_prompt function to optimise your LLM's responses. For more information about Langfuse: https://langfuse.com/guides/videos/introducing-langfuse-2.0

## Summary_prompt evaluation
The summary_prompt is evaluated using the RAGAS summarisation score. Important to note is that the summarisation score offered by RAGAS presents an error if the input length to the function that calculates the summary scores exceeds a token_length of 5000.
