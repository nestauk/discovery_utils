import os

from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import evaluate
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from transformers import PreTrainedTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import pipeline

from discovery_utils import logging
from discovery_utils.getters.horizon_scout import get_training_data
from discovery_utils.horizon_scout.utils import get_current_datetime
from discovery_utils.horizon_scout.utils import make_train_val_datasets
from discovery_utils.utils import s3


logging.basicConfig(level=logging.INFO)


def tokenize(df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """Preprocesse the text data in a DataFrame using a specified tokenizer.

    Args:
        df (pd.DataFrame): A DataFrame containing a column 'text' with text data to be tokenized.
        tokenizer (PreTrainedTokenizer): A tokenizer instance from the transformers library.

    Returns:
        Dict[str, Any]: A dictionary containing tokenized representations of the text data.
    """
    return tokenizer(df["text"], truncation=True, max_length=512)


def create_compute_metrics() -> Callable[[Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """
    Create a compute_metrics function that computes accuracy.

    Returns:
        Callable[[Tuple[np.ndarray, np.ndarray]], Dict[str, float]]: A function that computes accuracy
        for predictions and labels.
    """
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute accuracy for predictions and labels.

        Args:
            eval_pred (Tuple[np.ndarray, np.ndarray]): A tuple containing predictions and labels.

        Returns:
            Dict[str, float]: A dictionary with the accuracy score.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def train_and_save_model(
    mission: str, epochs: int, batch_size: int, id2label: Dict[int, str], label2id: Dict[str, int]
) -> None:
    """Train a model for a specific mission and save it to S3.

    Args:
        mission (str): The mission name.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        id2label (Dict[int, str]): Dictionary mapping label IDs to label names.
        label2id (Dict[str, int]): Dictionary mapping label names to label IDs.
    """
    logging.info("Starting %s training run", mission)
    # Make train and val dfs
    logging.info("Making train and validation dataframes")
    train_df, val_df = make_train_val_datasets(get_training_data(mission), 13, 0.80)

    # Make train and val HF datasets
    logging.info("Making train and validation Huggingface datasets")
    train_hf_ds = Dataset.from_pandas(train_df.assign(label=lambda x: x.relevant))
    val_hf_ds = Dataset.from_pandas(val_df.assign(label=lambda x: x.relevant))

    # Load tokenizer
    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    # Tokenize train and val HF datasets
    logging.info("Tokenizing datasets")
    tokenized_train_hf_ds = train_hf_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    tokenized_val_hf_ds = val_hf_ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    # Initialize the data collator
    logging.info("Initializing data collator")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load the pre-trained DistilBERT model
    logging.info("Loading pre-trained DistilBERT model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    # Retrieve the current date and time
    logging.info("Retrieving the current date and time")
    date_time = get_current_datetime()

    # Define the function to compute evaluation metrics during training.
    logging.info("Defining the function to compute evaluation metrics during training.")
    compute_metrics = create_compute_metrics()

    # Set training arguments
    logging.info("Setting training arguments")
    training_args = TrainingArguments(
        output_dir=os.path.expanduser(f"~/models_tmp/{mission}_distillbert_{date_time}"),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        push_to_hub=False,
    )

    # Train the model
    logging.info("Training the %s classifier", mission)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_hf_ds,
        eval_dataset=tokenized_val_hf_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0001)],
    )
    trainer.train()

    # Store number of steps
    global_step = trainer.state.global_step

    # Store eval_accuracy
    eval_results = trainer.evaluate()
    eval_accuracy = round(eval_results["eval_accuracy"], 3)

    # Load model that can be used for inference
    model_path = os.path.expanduser(f"~/models_tmp/{mission}_distillbert_{date_time}/checkpoint-{global_step}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=512, batch_size=batch_size
    )

    # Save model to S3
    save_path = f"models/horizon_scout/{mission}/{mission}_bert_{eval_accuracy}_{date_time}.pkl"
    logging.info("Saving the %s classifier to %s", mission, save_path)
    s3.upload_obj(
        classifier,
        s3.BUCKET_NAME_RAW,
        save_path,
    )


MISSIONS = ["ASF", "AFS", "AHL"]
EPOCHS = 10
ID2LABEL = {0: "NOT RELEVANT", 1: "RELEVANT"}
LABEL2ID = {"NOT RELEVANT": 0, "RELEVANT": 1}
BATCH_SIZE = 60

if __name__ == "__main__":
    for mission in MISSIONS:
        train_and_save_model(mission, EPOCHS, BATCH_SIZE, ID2LABEL, LABEL2ID)
