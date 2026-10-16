from pathlib import Path
import io

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UrbanKart Customer Intelligence Hub", page_icon="🛍️", layout="wide")
ROOT = Path(__file__).resolve().parent
REGRESSION_PATH = ROOT / "urbankart_best_regression_model.joblib"
REVIEWS_PATH = ROOT / "UrbanKart_HuggingFace_NLP_Customer_Reviews.csv"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVIEW_CANDIDATES = ("review_text", "review", "customer_review", "feedback", "text")


@st.cache_resource(show_spinner=False)
def load_regression_artifact(path: str, modified: float):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"model", "features", "model_name"}.issubset(artifact):
        raise ValueError("Model artifact must contain 'model', 'features', and 'model_name'.")
    return artifact


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline(model_name: str):
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


def find_review_column(columns):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in REVIEW_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    return next((c for c in columns if "review" in str(c).lower() or "feedback" in str(c).lower()), None)


def sentiment_label(value):
    value = str(value).strip().lower()
    return value.replace("label_", "label ") if value.startswith("label_") else value


def predict_reviews(classifier, texts, batch_size=16):
    predictions = []
    texts = [str(t).strip() for t in texts]
    for start in range(0, len(texts), batch_size):
        predictions.extend(classifier(texts[start:start + batch_size], truncation=True, max_length=512, batch_size=batch_size))
    return ([sentiment_label(p.get("label", "unknown")) for p in predictions],
            [float(p.get("score", 0.0)) for p in predictions])


def analyze_dataframe(df, review_col, classifier):
    result = df.copy()
    reviews = result[review_col].fillna("").astype(str).str.strip()
    valid = reviews.ne("")
    result["predicted_sentiment"] = "not analyzed (empty review)"
    result["sentiment_confidence"] = float("nan")
    if valid.any():
        labels, scores = predict_reviews(classifier, reviews.loc[valid].tolist())
        result.loc[valid, "predicted_sentiment"] = labels
        result.loc[valid, "sentiment_confidence"] = scores
    return result, int(valid.sum())


def revenue_page():
    st.title("🛍️ UrbanKart Customer Intelligence Hub")
    st.subheader("💰 Customer Revenue Prediction")
    st.write("Estimate annual customer revenue using profile, engagement, purchase, loyalty and service metrics.")
    st.divider()
    if not REGRESSION_PATH.exists():
        st.error(f"Model file not found: `{REGRESSION_PATH.name}`. Keep it in the same folder as `app.py`.")
        return
    try:
        artifact = load_regression_artifact(str(REGRESSION_PATH), REGRESSION_PATH.stat().st_mtime)
        model, features, model_name = artifact["model"], artifact["features"], artifact["model_name"]
    except Exception as exc:
        st.error(f"Could not load regression model: {exc}")
        st.info("Check the saved artifact format and install compatible scikit-learn/joblib versions.")
        return
    st.caption(f"Model: **{model_name}** · {len(features)} input features")
    with st.form("revenue_prediction_form"):
        st.header("Customer Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Annual Income (₹)", min_value=0.0, value=500000.0, step=10000.0)
            tenure = st.number_input("Customer Tenure (Months)", min_value=0, max_value=300, value=24)
            website_visits = st.number_input("Monthly Website Visits", min_value=0, max_value=1000, value=10)
            app_sessions = st.number_input("Monthly App Sessions", min_value=0, max_value=1000, value=10)
            purchase_frequency = st.number_input("Monthly Purchase Frequency", min_value=0.0, value=3.0, step=0.5)
        with c2:
            average_order_value = st.number_input("Average Order Value (₹)", min_value=0.0, value=2000.0, step=100.0)
            discount_usage_rate = st.slider("Discount Usage Rate", 0.0, 1.0, 0.2)
            orders_last_12m = st.number_input("Orders in Last 12 Months", min_value=0, max_value=1000, value=24)
            days_since_purchase = st.number_input("Days Since Last Purchase", min_value=0, max_value=3650, value=15)
            return_rate = st.slider("Return Rate", 0.0, 1.0, 0.05)
        with c3:
            support_tickets = st.number_input("Support Tickets in Last 12 Months", min_value=0, max_value=100, value=1)
            satisfaction = st.number_input("Customer Satisfaction Score", min_value=0.0, max_value=10.0, value=7.0)
            payment_delays = st.number_input("Payment Delay Count", min_value=0, max_value=100, value=0)
            loyalty_points = st.number_input("Loyalty Points", min_value=0, max_value=1000000, value=500)
            marketing_response = st.slider("Marketing Response Rate", 0.0, 1.0, 0.2)
        submitted = st.form_submit_button("🔮 Predict Annual Revenue", type="primary", use_container_width=True)
    if submitted:
        customer = {
            "Age": age, "Income": income, "Customer_Tenure_Months": tenure,
            "Website_Visits_Monthly": website_visits, "App_Sessions_Monthly": app_sessions,
            "Purchase_Frequency_Monthly": purchase_frequency, "Average_Order_Value": average_order_value,
            "Discount_Usage_Rate": discount_usage_rate, "Orders_Last_12M": orders_last_12m,
            "Days_Since_Last_Purchase": days_since_purchase, "Return_Rate": return_rate,
            "Support_Tickets_12M": support_tickets, "Customer_Satisfaction_Score": satisfaction,
            "Payment_Delay_Count": payment_delays, "Loyalty_Points": loyalty_points,
            "Marketing_Response_Rate": marketing_response,
        }
        missing = [feature for feature in features if feature not in customer]
        if missing:
            st.error(f"The input form does not provide these saved-model features: {missing}")
            return
        try:
            input_df = pd.DataFrame([customer]).reindex(columns=features)
            prediction = float(model.predict(input_df)[0])
            st.divider()
            st.header("📊 Prediction Result")
            m1, m2 = st.columns(2)
            m1.metric("Estimated Annual Revenue", f"₹{max(0.0, prediction):,.2f}")
            m2.metric("Model Used", str(model_name))
            if prediction < 0:
                st.warning("The model returned a negative estimate; the displayed value is floored at ₹0.")
            st.caption("This is a model estimate, not a guaranteed outcome. Actual customer revenue may differ.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    with st.expander("About this model"):
        st.write(f"**Model:** {model_name}")
        st.write("**Target:** `Annual_Revenue`")
        st.write(f"**Input features:** {len(features)}")


def feedback_page():
    st.title("💬 UrbanKart Customer Feedback Analysis")
    st.caption("Hugging Face Transformers · Three-class sentiment analysis")
    st.markdown("Analyze a single review or run batch analysis. Predictions are **negative**, **neutral**, or **positive** with a confidence score.")
    with st.sidebar:
        st.subheader("NLP model")
        st.code(SENTIMENT_MODEL, language=None)
        st.caption("The model downloads on first use; internet access is required unless cached.")
        st.caption("Confidence is a model score, not a guarantee of correctness.")
    st.divider()
    st.header("1. Dataset inspection")
    uploaded = st.file_uploader("Optional: upload a customer review CSV", type=["csv"], key="feedback_csv_uploader")
    dataset, source = None, None
    try:
        if uploaded is not None:
            data = uploaded.getvalue()
            source = uploaded.name
        elif REVIEWS_PATH.exists():
            data = REVIEWS_PATH.read_bytes()
            source = REVIEWS_PATH.name
        else:
            data = None
        if data is not None:
            dataset = read_csv_bytes(data)
    except Exception as exc:
        st.error(f"Could not read the review CSV: {exc}")
    review_col = find_review_column(dataset.columns) if dataset is not None else None
    if dataset is not None:
        a, b, c = st.columns(3)
        a.metric("Dataset rows", f"{len(dataset):,}")
        b.metric("Columns", f"{len(dataset.columns):,}")
        c.metric("Empty review cells", f"{dataset[review_col].isna().sum():,}" if review_col else "Unknown")
        st.write(f"**Loaded file:** `{source}`")
        if review_col:
            st.success(f"Review-text column identified: `{review_col}`")
        else:
            st.error("No review-text column found. Expected `review_text` or a similarly named column.")
        with st.expander("Preview first five rows"):
            st.dataframe(dataset.head(), use_container_width=True)
    else:
        st.warning(f"`{REVIEWS_PATH.name}` not found in the project root. Place it beside `app.py` or upload a CSV.")
    st.divider()
    st.header("2. Analyze a single review")
    review = st.text_area("Customer feedback", placeholder="The delivery was quick and the product quality was excellent.", height=120, key="single_review_text")
    if st.button("Analyze review", type="primary", key="analyze_single_review"):
        if not review.strip():
            st.warning("Enter a review before running sentiment analysis.")
        else:
            try:
                with st.spinner("Loading model and analyzing review..."):
                    classifier = load_sentiment_pipeline(SENTIMENT_MODEL)
                    result = classifier(review.strip(), truncation=True, max_length=512)[0]
                label, score = sentiment_label(result.get("label", "unknown")), float(result.get("score", 0.0))
                if label == "positive":
                    st.success(f"Predicted sentiment: **{label.title()}**")
                elif label == "negative":
                    st.error(f"Predicted sentiment: **{label.title()}**")
                else:
                    st.info(f"Predicted sentiment: **{label.title()}**")
                x, y = st.columns([2, 1])
                x.progress(max(0.0, min(1.0, score)), text="Model confidence")
                y.metric("Confidence score", f"{score:.1%}")
            except Exception as exc:
                st.error(f"Could not load or run the sentiment model: {exc}")
                st.info("Check internet access and confirm compatible `transformers` and `torch` are installed.")
    st.divider()
    st.header("3. Batch sentiment analysis")
    if dataset is not None and review_col is not None:
        if len(dataset) == 0:
            st.warning("The dataset has no rows to analyze.")
        else:
            row_limit = st.number_input("Rows to analyze (start small)", min_value=1, max_value=len(dataset), value=min(100, len(dataset)), step=100 if len(dataset) >= 100 else 1, key="feedback_row_limit")
            if st.button("Run batch sentiment analysis", key="run_feedback_batch", use_container_width=True):
                try:
                    selected = dataset.head(int(row_limit)).copy()
                    nonempty = selected[review_col].fillna("").astype(str).str.strip().ne("").sum()
                    if not nonempty:
                        st.warning("No non-empty reviews in the selected rows.")
                    else:
                        with st.spinner(f"Analyzing {nonempty:,} reviews in batches..."):
                            classifier = load_sentiment_pipeline(SENTIMENT_MODEL)
                            results, count = analyze_dataframe(selected, review_col, classifier)
                        st.session_state["urbankart_feedback_results"] = results
                        st.success(f"Analysis complete: {count:,} non-empty reviews analyzed.")
                except Exception as exc:
                    st.error(f"Batch analysis failed: {exc}")
                    st.info("Check model download/network access, memory, and installed dependencies.")
    results = st.session_state.get("urbankart_feedback_results")
    if isinstance(results, pd.DataFrame):
        st.subheader("Batch results")
        valid = results[results["predicted_sentiment"] != "not analyzed (empty review)"]
        if not valid.empty:
            distribution = valid["predicted_sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
            left, right = st.columns([1, 1.5])
            with left:
                st.markdown("**Sentiment distribution**")
                st.bar_chart(distribution.set_index("Sentiment"))
                percentages = valid["predicted_sentiment"].value_counts(normalize=True).mul(100).rename_axis("Sentiment").reset_index(name="Percentage")
                percentages["Percentage"] = percentages["Percentage"].map(lambda v: f"{v:.1f}%")
                st.dataframe(percentages, hide_index=True, use_container_width=True)
            with right:
                st.markdown("**Prediction preview**")
                st.dataframe(results.head(100), use_container_width=True)
        else:
            st.warning("No non-empty reviews were available to analyze.")
        st.download_button("Download analyzed reviews as CSV", data=results.to_csv(index=False).encode("utf-8-sig"), file_name="UrbanKart_Customer_Feedback_Sentiment_Results.csv", mime="text/csv", use_container_width=True)
    st.caption("This model was pretrained on Twitter text. Customer reviews may differ from its training domain; validate predictions before business use.")


pages = {
    "Customer Intelligence": [
        st.Page(revenue_page, title="Revenue Prediction", icon="💰", default=True),
        st.Page(feedback_page, title="Customer Feedback Analysis", icon="💬"),
    ]
}
st.navigation(pages).run()
