from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Feedback Analysis | UrbanKart",
    page_icon="💬",
    layout="wide",
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_PATH = Path(__file__).resolve().parent / "UrbanKart_HuggingFace_NLP_Customer_Reviews.csv"
REVIEW_COLUMN_CANDIDATES = [
    "review_text", "Review_Text", "review", "Review", "customer_review",
    "Customer_Review", "feedback", "Feedback", "text", "Text",
]

# -----------------------------
# Cached model and dataset loader
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline(model_name: str):
    """Load the Hugging Face sentiment pipeline once per Streamlit process."""
    from transformers import pipeline

    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
    )


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV from bytes so both local and uploaded files can be handled."""
    import io

    return pd.read_csv(io.BytesIO(file_bytes))


def find_review_column(columns):
    """Find the review-text column, preferring the dataset's known column name."""
    column_lookup = {str(column).strip().lower(): column for column in columns}
    for candidate in REVIEW_COLUMN_CANDIDATES:
        match = column_lookup.get(candidate.strip().lower())
        if match is not None:
            return match
    # Last-resort heuristic for similarly named columns.
    for column in columns:
        normalized = str(column).strip().lower()
        if "review" in normalized or "feedback" in normalized:
            return column
    return None


def analyze_texts(sentiment_pipeline, texts, batch_size=16):
    """Return predicted sentiment and confidence for each non-empty review."""
    predictions = []
    clean_texts = [str(text).strip() for text in texts]
    for start in range(0, len(clean_texts), batch_size):
        batch = clean_texts[start : start + batch_size]
        batch_predictions = sentiment_pipeline(
            batch,
            truncation=True,
            max_length=512,
            batch_size=batch_size,
        )
        predictions.extend(batch_predictions)

    labels = [str(item.get("label", "unknown")).replace("LABEL_", "Label ").lower() for item in predictions]
    scores = [float(item.get("score", 0.0)) for item in predictions]
    return labels, scores


def run_batch_analysis(df, review_column, sentiment_pipeline):
    """Analyze non-empty reviews while preserving the original rows."""
    result = df.copy()
    review_series = result[review_column].fillna("").astype(str).str.strip()
    valid_mask = review_series.ne("")

    result["predicted_sentiment"] = "not analyzed (empty review)"
    result["sentiment_confidence"] = pd.NA

    valid_texts = review_series.loc[valid_mask].tolist()
    if valid_texts:
        labels, scores = analyze_texts(sentiment_pipeline, valid_texts)
        result.loc[valid_mask, "predicted_sentiment"] = labels
        result.loc[valid_mask, "sentiment_confidence"] = scores

    result["sentiment_confidence"] = pd.to_numeric(
        result["sentiment_confidence"], errors="coerce"
    )
    return result, int(valid_mask.sum())


# -----------------------------
# UI
# -----------------------------
st.title("💬 UrbanKart Customer Feedback Analysis")
st.caption("Hugging Face NLP · Review sentiment classification")
st.markdown(
    "Analyze an individual customer review or run sentiment analysis across the UrbanKart review dataset. "
    "The model predicts **negative**, **neutral**, or **positive** sentiment and provides a confidence score."
)
st.divider()

with st.sidebar:
    st.header("Model information")
    st.write("**Pretrained model**")
    st.code(MODEL_NAME, language=None)
    st.caption(
        "The first run may take time because the pretrained model and tokenizer need to be downloaded. "
        "An internet connection is required on the first load."
    )
    st.caption("Confidence is the model's score for its predicted class; it is not a guarantee of correctness.")

# Load the dataset first and explicitly identify the review column.
st.header("1. Dataset inspection")
uploaded_file = st.file_uploader("Optional: upload a review CSV", type=["csv"])

try:
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        source_name = uploaded_file.name
    elif DATASET_PATH.exists():
        raw_bytes = DATASET_PATH.read_bytes()
        source_name = DATASET_PATH.name
    else:
        raw_bytes = None
        source_name = None

    dataset = load_csv(raw_bytes) if raw_bytes is not None else None
except Exception as exc:
    dataset = None
    st.error(f"Could not read the CSV file: {exc}")
    st.info("Check that the file is a valid CSV encoded in a format pandas can read.")

review_column = None
if dataset is not None:
    review_column = find_review_column(dataset.columns)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Dataset rows", f"{len(dataset):,}")
    metric2.metric("Columns", f"{len(dataset.columns):,}")
    metric3.metric("Empty review cells", f"{dataset[review_column].isna().sum():,}" if review_column else "Unknown")
    st.write(f"**Loaded file:** `{source_name}`")
    if review_column:
        st.success(f"Review-text column identified: `{review_column}`")
    else:
        st.error("Could not identify a review-text column. Expected a column such as `review_text` or `feedback`.")
    with st.expander("Preview first 5 rows"):
        st.dataframe(dataset.head(), use_container_width=True)
else:
    st.warning(
        "The default dataset was not found. Place `UrbanKart_HuggingFace_NLP_Customer_Reviews.csv` "
        "beside this Python file, or upload the CSV above. The single-review analyzer can still be used."
    )

st.divider()
st.header("2. Analyze a single review")
review_input = st.text_area(
    "Enter customer feedback",
    placeholder="Example: The delivery was quick and the product quality was excellent.",
    height=120,
)

single_col1, single_col2 = st.columns([1, 3])
with single_col1:
    analyze_single = st.button("Analyze review", type="primary", use_container_width=True)

if analyze_single:
    if not review_input.strip():
        st.warning("Enter a review before running sentiment analysis.")
    else:
        try:
            with st.spinner("Loading model and analyzing review…"):
                classifier = load_sentiment_pipeline(MODEL_NAME)
                single_result = classifier(
                    review_input.strip(), truncation=True, max_length=512
                )[0]
            label = str(single_result.get("label", "unknown")).replace("LABEL_", "Label ").lower()
            confidence = float(single_result.get("score", 0.0))
            pretty_label = label.title()
            if label == "positive":
                st.success(f"Predicted sentiment: **{pretty_label}**")
            elif label == "negative":
                st.error(f"Predicted sentiment: **{pretty_label}**")
            else:
                st.info(f"Predicted sentiment: **{pretty_label}**")
            st.progress(max(0.0, min(1.0, confidence)))
            st.metric("Confidence score", f"{confidence:.1%}")
        except Exception as exc:
            st.error(f"Could not load or run the sentiment model: {exc}")
            st.info(
                "Check your internet connection and confirm that `transformers` and `torch` are installed. "
                "If the model download fails, retry after connectivity is restored."
            )

st.divider()
st.header("3. Batch sentiment analysis")
if dataset is not None and review_column is not None:
    st.write(f"Batch analysis will use the identified `{review_column}` column.")
    max_rows = len(dataset)
    row_limit = st.number_input(
        "Rows to analyze (use a smaller number for a quick first test)",
        min_value=1,
        max_value=max(1, max_rows),
        value=min(100, max(1, max_rows)),
        step=100 if max_rows >= 100 else 1,
    )
    if st.button("Run batch sentiment analysis", use_container_width=True):
        try:
            batch_df = dataset.head(int(row_limit)).copy()
            with st.spinner("Loading model and analyzing reviews in batches…"):
                classifier = load_sentiment_pipeline(MODEL_NAME)
                analyzed_df, analyzed_count = run_batch_analysis(
                    batch_df, review_column, classifier
                )
            st.session_state["urban_kart_sentiment_results"] = analyzed_df
            st.session_state["urban_kart_sentiment_source"] = source_name or "uploaded CSV"
            st.success(f"Analysis complete: {analyzed_count:,} non-empty reviews analyzed.")
        except Exception as exc:
            st.error(f"Batch analysis failed: {exc}")
            st.info(
                "Check model download/network access, available memory, and that `transformers` and `torch` are installed."
            )
elif dataset is not None:
    st.info("Batch analysis is unavailable until a review-text column can be identified.")
else:
    st.info("Upload or place the dataset beside this file to enable batch analysis.")

results = st.session_state.get("urban_kart_sentiment_results")
if isinstance(results, pd.DataFrame):
    st.subheader("Batch results")
    valid_results = results[results["predicted_sentiment"] != "not analyzed (empty review)"]
    if not valid_results.empty:
        distribution = (
            valid_results["predicted_sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )
        dist_col, table_col = st.columns([1, 1.5])
        with dist_col:
            st.markdown("**Sentiment distribution**")
            st.bar_chart(distribution.set_index("Sentiment"))
            percentages = (
                valid_results["predicted_sentiment"].value_counts(normalize=True).mul(100)
            )
            st.dataframe(
                percentages.rename("Percentage").reset_index().rename(
                    columns={"index": "Sentiment", "predicted_sentiment": "Sentiment"}
                ).assign(Percentage=lambda frame: frame["Percentage"].map(lambda x: f"{x:.1f}%")),
                hide_index=True,
                use_container_width=True,
            )
        with table_col:
            st.markdown("**Predictions preview**")
            st.dataframe(results.head(100), use_container_width=True)
    else:
        st.warning("No non-empty reviews were available to analyze.")

    csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download analyzed reviews as CSV",
        data=csv_bytes,
        file_name="UrbanKart_Customer_Feedback_Sentiment_Results.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "Note: This model was pretrained on Twitter text. Customer reviews may differ from its training domain, "
    "so validate predictions before using them for business decisions."
)
