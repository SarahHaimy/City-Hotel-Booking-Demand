# Live Discussion Demonstration Guide (10-Minute Walkthrough)

## Pre-Discussion Checklist
1. Activate virtual environment: `source .venv/bin/activate`
2. Verify all official artifacts are present locally:
   ```bash
   python main.py evaluate --task regression --model all
   python main.py evaluate --task classification --model all
   ```
   *(If cloning fresh from the submitted ZIP where `.joblib` files are excluded per Spec §17, run `python main.py build-artifacts` once before the demo.)*
3. Launch the Streamlit ML Studio:
   ```bash
   python main.py gui
   ```

---

## Step-by-Step Demo Flow (Covers Every PDF §13 & §19 Requirement)

1. **Home & Deployed Models (1 min)**
   - Show the 3 active deployed pipelines (ADR Regression, Cancellation Classification, Guest Segmentation), their validation + unseen test metrics, selection criteria, and the green sidebar health badge proving zero startup retraining.
2. **Data Explorer & Anti-Leakage Audit (1.5 min)**
   - Open **Data Explorer**: show raw H2 shape (`79,330 × 31`), `clean_data()` step-by-step row removal audit, `Babies → HasBabies` mapping, and stratified 70/15/15 splits.
   - Open **Anti-Leakage Audit**: show live green `PASS` assertions and explain why `ReservationStatus`, `AssignedRoomType`, `BookingChanges`, `DaysInWaitingList`, and `DepositType` (for cancellation/clustering) are excluded at reservation time.
3. **Model Catalog & Specs (1.5 min)**
   - Browse **Model Catalog** across Regression, Classification, and Clustering.
   - Click a model card to inspect its pipeline diagram, baseline vs tuned hyperparameters, CV fold boxplot, parity/residual plots (Regression), and interactive threshold slider + MarketSegment false-alarm audit (Classification).
4. **Compare Models & Tuning Lab (2 min)**
   - Open **Compare Models**: show the executive text summary, color-highlighted multi-metric leaderboard, baseline-to-tuned dumbbell chart, training-time vs accuracy scatter, and overlaid ROC / PR / calibration curves.
   - Open **Hyperparameter Tuning Lab**: show documented search spaces, parallel-coordinates trial plots, and best configurations.
5. **Clustering & PCA/LDA Analysis (1.5 min)**
   - Open **Clustering & Personas**: compare K-Means, GMM, Agglomerative Hierarchical (with Ward dendrogram and documented 6,000-row representative sampling), and DBSCAN; walk through the 2D PCA projection and guest persona cards.
   - Open **PCA & LDA Analysis**: justify the 90% variance threshold (`7` PCs for clustering, `19` PCs for classification) and compare Original vs PCA vs 1D LDA projections.
6. **Live Inference, Batch Evaluation & Custom Training (2.5 min)**
   - Open **Predict Single Booking**: pick presets or adjust `LeadTime`, view instant ADR + cancellation risk + cluster persona, cross-model consensus table, and What-If sensitivity curves.
   - Open **Batch Predict & Evaluate**: click **📄 Labeled (250)** (or upload a CSV) to run batch predictions and ground-truth evaluation with shift detection.
   - Open **Train Studio**: click **📄 Labeled (250)**, pick any model, click **🚀 Launch Training Job**, watch the non-blocking progress bar & logs, and show the newly persisted user bundle in `artifacts/user/`!
