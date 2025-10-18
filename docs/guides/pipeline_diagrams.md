# EEG2025 Challenge Pipeline Diagrams

## Complete Training and Submission Pipeline

```mermaid
graph TB
    subgraph "Data Preparation"
        A[EEG Data] --> B[BIDS Structure]
        B --> C[StarterKit DataLoader]
        C --> D[Train/Val/Test Splits]
    end

    subgraph "SSL Pretraining"
        D --> E[SSL Augmentations]
        E --> F[Masked Time Reconstruction]
        E --> G[Contrastive Learning]
        E --> H[Predictive Residual]
        F --> I[SSL Model Training]
        G --> I
        H --> I
        I --> J[Pretrained Backbone]
    end

    subgraph "Cross-Task Transfer"
        J --> K[Cross-Task Model]
        D --> K
        K --> L[CCD Task Training]
        L --> M[RT Regression + Success Classification]
        M --> N[Cross-Task Predictions]
    end

    subgraph "Psychopathology with DANN"
        J --> O[DANN Model]
        D --> O
        O --> P[Domain Adversarial Training]
        P --> Q[Gradient Reversal Layer]
        Q --> R[Task Head: CBCL Factors]
        Q --> S[Domain Head: Site Classification]
        R --> T[Uncertainty Weighted Loss]
        S --> U[Domain Adversarial Loss]
        T --> V[Psychopathology Predictions]
        U --> V
    end

    subgraph "Submission Generation"
        N --> W[Cross-Task CSV]
        V --> X[Psychopathology CSV]
        W --> Y[Schema Validation]
        X --> Y
        Y --> Z[Submission Archive]
        Z --> AA[Challenge Submission]
    end

    style A fill:#e1f5fe
    style J fill:#c8e6c9
    style N fill:#fff3e0
    style V fill:#f3e5f5
    style AA fill:#ffebee
```

## SSL Pretraining Architecture

```mermaid
graph LR
    subgraph "Input Processing"
        A[Raw EEG<br/>64×1000] --> B[SSL Augmentations]
        B --> C[View 1]
        B --> D[View 2]
    end

    subgraph "SSL Backbone"
        C --> E[TemporalCNN<br/>Backbone]
        D --> F[TemporalCNN<br/>Backbone]
        E --> G[Features<br/>128-dim]
        F --> H[Features<br/>128-dim]
    end

    subgraph "SSL Objectives"
        G --> I[Masked Reconstruction<br/>Decoder]
        G --> J[Contrastive<br/>Projection Head]
        H --> J
        G --> K[Predictive<br/>Residual Head]
    end

    subgraph "Loss Computation"
        I --> L[Reconstruction Loss]
        J --> M[InfoNCE Loss]
        K --> N[Prediction Loss]
        L --> O[Combined SSL Loss]
        M --> O
        N --> O
    end

    style A fill:#e3f2fd
    style O fill:#e8f5e8
```

## DANN Domain Adversarial Training

```mermaid
graph TB
    subgraph "Input"
        A[EEG Data<br/>Multiple Sites] --> B[Backbone Network]
    end

    subgraph "Feature Extraction"
        B --> C[Shared Features<br/>F(x)]
    end

    subgraph "Task Branch"
        C --> D[Task Predictor<br/>Gₜ(F(x))]
        D --> E[CBCL Factors<br/>p_factor, int, ext, att]
    end

    subgraph "Domain Branch"
        C --> F[Gradient Reversal<br/>Layer (λ schedule)]
        F --> G[Domain Classifier<br/>Gd(F(x))]
        G --> H[Site Predictions]
    end

    subgraph "Loss Components"
        E --> I[Task Loss<br/>Uncertainty Weighted]
        H --> J[Domain Loss<br/>Cross-Entropy]
        I --> K[Total Loss]
        J --> K
    end

    subgraph "Lambda Scheduling"
        L[Training Step] --> M[GRL Scheduler<br/>λ: 0→0.2]
        M --> F
    end

    style C fill:#fff3e0
    style F fill:#ffebee
    style K fill:#e8f5e8
```

## Uncertainty Weighted Multi-Task Learning

```mermaid
graph LR
    subgraph "Task Predictions"
        A[Shared Features] --> B[p_factor Head]
        A --> C[Internalizing Head]
        A --> D[Externalizing Head]
        A --> E[Attention Head]
    end

    subgraph "Uncertainty Parameters"
        F[log σ₁²] --> G[σ₁² = exp(log σ₁²)]
        H[log σ₂²] --> I[σ₂² = exp(log σ₂²)]
        J[log σ₃²] --> K[σ₃² = exp(log σ₃²)]
        L[log σ₄²] --> M[σ₄² = exp(log σ₄²)]
    end

    subgraph "Weighted Loss"
        B --> N[L₁/(2σ₁²) + log σ₁]
        C --> O[L₂/(2σ₂²) + log σ₂]
        D --> P[L₃/(2σ₃²) + log σ₃]
        E --> Q[L₄/(2σ₄²) + log σ₄]

        N --> R[Combined Loss]
        O --> R
        P --> R
        Q --> R
    end

    style G fill:#e1f5fe
    style R fill:#e8f5e8
```

## Submission Pipeline Flow

```mermaid
graph TB
    subgraph "Model Outputs"
        A[Cross-Task Model] --> B[RT Predictions]
        A --> C[Success Probabilities]
        D[Psychopathology Model] --> E[CBCL Factor Scores]
    end

    subgraph "Data Validation"
        B --> F[Validate RT Range]
        C --> G[Validate Probabilities]
        E --> H[Validate CBCL Scores]
    end

    subgraph "CSV Generation"
        F --> I[cross_task_submission.csv]
        G --> I
        H --> J[psychopathology_submission.csv]
    end

    subgraph "Schema Validation"
        I --> K[Cross-Task Schema Check]
        J --> L[Psych Schema Check]
        K --> M{All Valid?}
        L --> M
    end

    subgraph "Archive Creation"
        M -->|Yes| N[Create Submission Manifest]
        N --> O[Package ZIP Archive]
        O --> P[Final Submission]
        M -->|No| Q[Validation Errors]
    end

    style P fill:#c8e6c9
    style Q fill:#ffcdd2
```

## Configuration Flow

```mermaid
graph LR
    subgraph "Config Files"
        A[pretrain.yaml] --> B[SSL Training]
        C[train_cross_task.yaml] --> D[Cross-Task Training]
        E[train_psych.yaml] --> F[DANN Training]
    end

    subgraph "Shared Components"
        B --> G[Pretrained Backbone]
        G --> D
        G --> F
    end

    subgraph "Model Outputs"
        D --> H[Cross-Task Checkpoints]
        F --> I[Psychopathology Checkpoints]
    end

    subgraph "Evaluation"
        H --> J[Cross-Task Metrics]
        I --> K[Psych Metrics]
        J --> L[Submission Generation]
        K --> L
    end

    style G fill:#fff3e0
    style L fill:#e8f5e8
```

## GRL Lambda Scheduling Strategies

```mermaid
graph TB
    subgraph "Linear Warmup"
        A[Step 0<br/>λ = 0.0] --> B[Step 500<br/>λ = 0.1]
        B --> C[Step 1000<br/>λ = 0.2]
        C --> D[Step 1000+<br/>λ = 0.2]
    end

    subgraph "Exponential Decay"
        E[λ(t) = λ_final × (1 - exp(-γt))]
        E --> F[Smooth Exponential Curve]
    end

    subgraph "Cosine Annealing"
        G[Warmup Phase<br/>Linear 0→λ_max]
        G --> H[Cosine Decay<br/>λ_max→λ_min]
    end

    subgraph "Adaptive"
        I[Domain Accuracy] --> J{High Accuracy?}
        J -->|Yes| K[Increase λ]
        J -->|No| L[Decrease λ]
        K --> M[Update λ]
        L --> M
    end

    style C fill:#c8e6c9
    style M fill:#fff3e0
```

## Data Flow Architecture

```mermaid
graph TB
    subgraph "Data Loading"
        A[BIDS Dataset] --> B[StarterKit Loader]
        B --> C[Subject Splits]
        C --> D[Session Batching]
    end

    subgraph "Preprocessing"
        D --> E[Channel Selection]
        E --> F[Temporal Windowing]
        F --> G[Normalization]
    end

    subgraph "Augmentation Pipeline"
        G --> H[Time Masking]
        H --> I[Channel Dropout]
        I --> J[Temporal Jitter]
        J --> K[Frequency Masking]
        K --> L[Augmented Views]
    end

    subgraph "Model Training"
        L --> M[Batch Processing]
        M --> N[Forward Pass]
        N --> O[Loss Computation]
        O --> P[Backpropagation]
        P --> Q[Parameter Update]
    end

    style L fill:#e1f5fe
    style Q fill:#e8f5e8
```

## Testing Architecture

```mermaid
graph LR
    subgraph "Unit Tests"
        A[DANN Components] --> B[GRL Layer Tests]
        A --> C[Scheduler Tests]
        A --> D[Domain Head Tests]
    end

    subgraph "Integration Tests"
        E[End-to-End Pipeline] --> F[SSL→Transfer Flow]
        E --> G[DANN Training Flow]
        E --> H[Submission Generation]
    end

    subgraph "Validation Tests"
        I[Schema Compliance] --> J[CSV Format Tests]
        I --> K[Archive Structure Tests]
        I --> L[Metadata Validation]
    end

    subgraph "Performance Tests"
        M[Memory Profiling] --> N[GPU Usage Tests]
        M --> O[Speed Benchmarks]
        M --> P[Scaling Tests]
    end

    style F fill:#e8f5e8
    style J fill:#fff3e0
    style N fill:#e1f5fe
```

## Reproducibility Pipeline

```mermaid
graph TB
    subgraph "Environment Capture"
        A[System Info] --> B[Python Version]
        A --> C[PyTorch Version]
        A --> D[CUDA Version]
        A --> E[Package List]
        A --> F[Git Commit]
    end

    subgraph "Seed Management"
        G[Base Seed] --> H[Component Seeds]
        H --> I[Python Random]
        H --> J[NumPy Random]
        H --> K[PyTorch Random]
        H --> L[CUDA Random]
    end

    subgraph "Run Tracking"
        M[Experiment Start] --> N[Config Snapshot]
        N --> O[Data Hash]
        O --> P[Training Metrics]
        P --> Q[Model Checkpoints]
        Q --> R[Run Manifest]
    end

    subgraph "Validation"
        R --> S[Reproduce Results]
        S --> T[Compare Outputs]
        T --> U[Verify Determinism]
    end

    style R fill:#c8e6c9
    style U fill:#e8f5e8
```

This comprehensive set of diagrams visualizes the complete EEG2025 challenge pipeline from data loading through final submission, including all major components like SSL pretraining, DANN domain adaptation, and submission validation.
