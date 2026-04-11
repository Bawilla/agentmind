# MLOps Pipeline — CV Bullet Points

Use these 5 bullet points in your CV / LinkedIn under the AgentMind project.

---

- **Engineered a 4-stage GitHub Actions CI/CD pipeline** (test → build → push → deploy)
  that automatically containerises the RAG service, pushes to AWS ECR, and force-deploys
  to AWS ECS Fargate on every push to `main`, reducing manual deployment effort to zero.

- **Instrumented production RAG inference with MLflow experiment tracking**, logging
  10 metrics per request (latency, chunk grades, tool usage, answer length) to the
  `agentmind-rag` experiment; enables per-commit performance comparison and regression
  detection across hundreds of runs via the MLflow UI.

- **Built an Evidently AI drift monitoring system** that appends every `/ask` query to
  a rolling JSONL log and auto-generates HTML `DataDriftPreset` reports every 50 queries,
  alerting when query-length, answer-length, or latency drift exceeds a 0.3 threshold —
  providing early warning of data distribution shifts in production.

- **Developed a real-time terminal monitoring dashboard** that reads the query log and
  surfaces daily query volume, average latency, tool-usage breakdown (retrieval vs
  web search), and top-5 question topics via keyword frequency analysis — zero external
  dependencies.

- **Integrated AWS SageMaker Experiment Tracking** into the FastAPI inference path,
  logging model configuration parameters and per-request performance metrics
  (`latency_ms`, `chunks_retrieved`, `answer_relevancy`) as SageMaker Runs under the
  `agentmind-production` experiment, enabling full production-run auditing and
  hyperparameter comparison in SageMaker Studio.
