# References

All 17 entries below were verified by live lookup (not recalled from
training data alone) before being added here -- each has a source URL
confirming author list, venue, year and (where applicable) DOI/volume/page
numbers. Numbering follows IEEE style for direct use in the paper.

1. J. Anikannal, "Solar Power Generation Data," Kaggle dataset, 2020.
   [Online]. Available: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
   -- *Primary dataset used throughout this work (Plant 1 and Plant 2
   generation and weather telemetry).*

2. International Electrotechnical Commission, *IEC 61724-1:2017,
   Photovoltaic system performance -- Part 1: Monitoring*, IEC, Geneva,
   Switzerland, 2017. [Online]. Available:
   https://webstore.iec.ch/en/publication/5733
   -- *Industry standard for PV performance monitoring; motivates the
   performance-ratio framing used for underperformance detection.*

3. J. Antonanzas, N. Osorio, R. Escobar, R. Urraca, F. J. Martinez-de-Pison,
   and F. Antonanzas-Torres, "Review of photovoltaic power forecasting,"
   *Solar Energy*, vol. 136, pp. 78-111, 2016, doi:
   10.1016/j.solener.2016.06.069.
   -- *Survey of ML methods for PV power forecasting; frames this paper's
   regression benchmark relative to the field.*

4. J. Francisti, K. Fodor, Z. Balogh, and M. Magdin, "Predictive modeling
   and anomaly detection in solar PV inverters using machine learning,"
   *ScienceDirect* (open-access preprint/journal article), 2025. [Online].
   Available: https://www.sciencedirect.com/science/article/pii/S2590123025043920
   -- *Directly comparable applied work on ML-based PV inverter anomaly
   detection; closest related work for the underperformance-detection
   contribution.*

5. V. Khandeparkar, Shreshtha, and S. K. Ramu, "Effectiveness of supervised
   machine learning models for electrical fault detection in solar PV
   systems," *Scientific Reports*, vol. 15, art. 34919, 2025, doi:
   10.1038/s41598-025-18802-4.
   -- *Recent supervised-ML PV fault-detection benchmark (RF, XGBoost,
   SVM, etc.); related-work comparison point for the model roster used
   here.*

6. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, pp. 5-32,
   2001, doi: 10.1023/A:1010933404324.
   -- *Random forest model used as the primary expected-output and
   interpretability/uncertainty reference model.*

7. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in
   *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data
   Mining (KDD)*, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.
   -- *XGBoost regressor, one of the two best-performing tabular models.*

8. G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu,
   "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in
   *Advances in Neural Information Processing Systems 30 (NIPS 2017)*,
   2017, pp. 3146-3154.
   -- *LightGBM regressor, tied with XGBoost for best tabular accuracy.*

9. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model
   Predictions," in *Advances in Neural Information Processing Systems 30
   (NIPS 2017)*, 2017. [Online]. Available:
   https://arxiv.org/abs/1705.07874
   -- *SHAP method used for the interpretability check against known
   solar-generation physics.*

10. N. Meinshausen, "Quantile Regression Forests," *Journal of Machine
    Learning Research*, vol. 7, pp. 983-999, 2006. [Online]. Available:
    https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    -- *Basis for the tree-quantile prediction interval used in the
    uncertainty calibration study.*

11. A. N. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal
    Prediction and Distribution-Free Uncertainty Quantification," arXiv
    preprint arXiv:2107.07511, 2021; published as "Conformal Prediction:
    A Gentle Introduction," *Foundations and Trends in Machine Learning*,
    vol. 16, no. 4, pp. 494-591, 2023.
    -- *Cited as the recommended fix for the uncertainty-interval
    miscalibration finding (future work).*

12. J. Quinonero-Candela, M. Sugiyama, A. Schwaighofer, and N. D. Lawrence,
    Eds., *Dataset Shift in Machine Learning*. Cambridge, MA, USA: MIT
    Press, 2009.
    -- *Theoretical framing for the cross-plant generalization / domain-shift
    finding.*

13. X. He, K. Zhao, and X. Chu, "AutoML: A survey of the state-of-the-art,"
    *Knowledge-Based Systems*, vol. 212, art. 106622, 2021, doi:
    10.1016/j.knosys.2020.106622.
    -- *Context for the managed-AutoML comparison against Vertex AI AutoML
    Tables.*

14. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
    Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J.
    VanderPlas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E.
    Duchesnay, "Scikit-learn: Machine Learning in Python," *Journal of
    Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
    -- *Software used for all classical/ensemble models, splits, and
    metrics.*

15. A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T.
    Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E.
    Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L.
    Fang, J. Bai, and S. Chintala, "PyTorch: An Imperative Style,
    High-Performance Deep Learning Library," in *Advances in Neural
    Information Processing Systems 32 (NeurIPS 2019)*, 2019, pp.
    8024-8035.
    -- *Software used for the MLP and 1D-CNN models.*

16. Google Cloud, "AutoML on Vertex AI -- Tabular data," Google Cloud
    documentation. [Online]. Available:
    https://cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/introduction
    -- *Reference for the managed AutoML Tables service used as the
    proprietary-baseline comparison.*

17. B. Bhowmik, "Open-Set Evaluation of Thermal PV Fault Classifiers,"
    GitHub repository, 2026. [Online]. Available:
    https://github.com/barnadeepb/open-set-solar-fault-detection
    -- *Author's own related but distinct prior work (thermal-image fault
    classification), cited to explicitly delineate scope and preempt any
    overlap concern with the present telemetry-based regression work.*

## Notes on remaining verification before submission

- Reference 4 (Francisti et al.) shows as a 2025 SSRN/ScienceDirect entry
  in search results without a fully confirmed final journal
  volume/page/issue at the time of this check -- re-verify the final
  published citation details (or DOI) directly on the publisher page
  before the camera-ready deadline, since preprint-to-final-journal
  metadata can change.
- Reference 16 is vendor documentation, not a peer-reviewed source --
  acceptable as a tool citation but should not be the sole citation
  standing in for a technical claim about AutoML in general (reference 13
  covers that).
- Double-check IEEE reference formatting/order against the actual PESA
  paper template once downloaded, since exact required style (numbering,
  "et al." thresholds, DOI vs. URL) is set by the template, not assumed
  here.
