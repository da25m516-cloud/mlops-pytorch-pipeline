# System Architecture

```mermaid
flowchart TD
    A[Developer] --> B[GitHub Repository]
    B --> C[Feature Branches and Pull Requests]
    C --> D[Develop and Main]

    E[CIFAR-10] --> F[PyTorch ResNet-18]
    F --> G[Docker GPU Training]
    G --> H[classifier_v1.pt]

    H --> I[PersistentVolumeClaim]
    J[ConfigMap] --> K[Kubernetes Job]
    I --> K

    I --> L[FastAPI Serving Deployment]
    L --> M[Two Serving Replicas]
    M --> N[Kubernetes Service]
    N --> O[Health and Prediction Requests]
    M --> P[HPA: 2 to 4 replicas]

