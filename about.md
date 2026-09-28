AeroTwin AI: Post-Flight Structural Crack Mapping and Seismic Vulnerability Assessment Using UAV Imagery

Description: AeroTwin AI is an intelligent structural health monitoring framework designed to streamline post-disaster building inspections using unmanned aerial vehicles (UAVs). By processing post-flight drone imagery, the system automatically segments structural materials, tags individual load-bearing columns, identifies surface defects such as micro-cracks, and computes a data-driven seismic vulnerability score to estimate building durability across various earthquake magnitudes.

Statement of the Problem:
This study aims to develop AeroTwin AI, a post-flight UAV-based framework for structural crack mapping and seismic vulnerability assessment of critical government facilities. Specifically, it seeks to:

1.)Develop and implement a post-flight UAV imagery processing pipeline that automatically segments structural materials, identifies load-bearing elements, and maps surface defects (micro-cracks and macro-cracks) on selected critical government facilities.
2.)Compute a data-driven seismic vulnerability score for each assessed building by integrating mapped structural defect metrics with relevant building profile data (e.g., structure age, construction type, and compliance indicators), resulting in a standardized vulnerability classification.
3.)Generate evidence-based recommendations on structural prioritization and potential intervention needs for the assessed critical facilities based on the vulnerability assessment results.

Project file structure: 
AeroTwin-AI/
│
├── backend/
│   │
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── processing.py
│   │   │   ├── assessment.py
│   │   │   └── reports.py
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   └── database.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── material_segmenter.py
│   │   │   ├── structural_element_detector.py
│   │   │   ├── crack_detector.py
│   │   │   └── weights/
│   │   │       ├── material_model.pt
│   │   │       ├── crack_model.pt
│   │   │       └── ...
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── uav_preprocessor.py
│   │   │   ├── crack_mapper.py
│   │   │   ├── defect_metrics.py
│   │   │   ├── seismic_calculator.py
│   │   │   └── report_generator.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── processing.py
│   │   │   ├── assessment.py
│   │   │   └── reports.py
│   │   │
│   │   └── main.py
│   │
│   ├── tests/
│   │   ├── test_preprocessing.py
│   │   ├── test_material_segmentation.py
│   │   ├── test_crack_detection.py
│   │   ├── test_crack_mapping.py
│   │   └── test_seismic_calculator.py
│   │
│   ├── requirements.txt
│   └── run_backend.py
│
├── frontend/
│   │
│   ├── public/
│   │
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── MaterialMap.jsx
│   │   │   ├── CrackViewer.jsx
│   │   │   ├── StructuralElementViewer.jsx
│   │   │   ├── ScoreCard.jsx
│   │   │   └── VulnerabilityBadge.jsx
│   │   │
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── InspectionView.jsx
│   │   │   ├── AssessmentView.jsx
│   │   │   └── ReportView.jsx
│   │   │
│   │   ├── services/
│   │   │   └── api.js
│   │   │
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── inputs/
│   │   ├── raw_images/
│   │   └── orthomosaics/
│   │
│   ├── processed/
│   │   ├── preprocessed/
│   │   ├── material_masks/
│   │   ├── structural_elements/
│   │   └── crack_maps/
│   │
│   ├── outputs/
│   │   ├── assessments/
│   │   ├── reports/
│   │   └── visualizations/
│   │
│   └── aerotwin.db
│
├── electron/
│   ├── main.js
│   └── preload.js
│
├── assets/
│   ├── icons/
│   └── installers/
│
├── docs/
│   ├── architecture/
│   ├── methodology/
│   └── sample_outputs/
│
├── README.md
└── package.json


Conceptual Architecture: 

                    AEROTWIN AI
                         │
              ┌──────────┴──────────┐
              │                     │
          FRONTEND                BACKEND
              │                     │
        React / Vite           FastAPI
              │                     │
              │              ┌──────┴──────┐
              │              │             │
              │           ML MODELS      SERVICES
              │              │             │
              │       ┌──────┼──────┐   ┌──┼──────────────┐
              │       │      │      │   │  │       │      │
              │   Material  Structural Crack UAV  Seismic Report
              │   Segment.  Elements Detector Prep Calculator Generator
              │                       │
              │                       ↓
              │                  Crack Mapping
              │                       ↓
              │                  Defect Metrics
              │                       ↓
              └──────────────→ Vulnerability Assessment
                                      │
                                      ↓
                              Classification
                                      │
                                      ↓
                              Recommendations
                                      │
                                      ↓
                                  Reports


