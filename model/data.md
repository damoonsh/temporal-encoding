
# Datasets

## Cryptocurrency Price History
[Cryptocurrency Price History](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory)
- Contains historical price data for various cryptocurrencies
- Includes different cryptocurrency datasets with price, volume, and market data
- Used for training and evaluation of cryptocurrency price prediction models

## Stock Exchange Data
[Index Data](https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data)

Filtered and cleaned data:
- NYA: Dropped missing values in Open and filtered based on index being NYA
- IHGN: Dropped missing values in Open and filtered based on index being in ['IXIC', 'HSI', 'GSPTSE', 'NSEI']
- GDXI: Dropped missing values in Open and filtered based on index being GDXI

## Data Patching Mechanism

The IndexData class combines multiple database sources, normalizes during filtering, then creates overlapping patches:

### 1. Input Sequence Structure
*Shows how time series data is organized before patching*

```
Original Sequence (seq_len = 12):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ v1  │ v2  │ v3  │ v4  │ v5  │ v6  │ v7  │ v8  │ v9  │ v10 │ v11 │ v12 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### 2. Overlapping Patch Creation
*Demonstrates how patches are extracted with stride < patch_length to create overlaps*

```
Parameters: patch_length = 4, num_patches = 4, patch_stride = 3 (overlapping)

Overlapping Patch Extraction:
┌─────┬─────┬─────┬─────┐                                                  
│ v1  │ v2  │ v3  │ v4  │ ← Patch 1 (indices 0-4)                        
└─────┴─────┴─────┴─────┘                                                  
          ┌─────┬─────┬─────┬─────┐                                        
          │ v4  │ v5  │ v6  │ v7  │ ← Patch 2 (indices 3-7) [OVERLAP]    
          └─────┴─────┴─────┴─────┘                                        
                    ┌─────┬─────┬─────┬─────┐                              
                    │ v7  │ v8  │ v9  │ v10 │ ← Patch 3 (indices 6-10) [OVERLAP]
                    └─────┴─────┴─────┴─────┘                              
                              ┌─────┬─────┬─────┬─────┐                    
                              │ v10 │ v11 │ v12 │ v13 │ ← Patch 4 (indices 9-13) [OVERLAP]
                              └─────┴─────┴─────┴─────┘                    
```

### 3. Final Tensor Output
*Shows the stacked patch tensor structure fed to the model*

```
Final Output Tensor [4, 4]:
┌─────────────────────┐
│ Patch 1: [v1-v4]    │
├─────────────────────┤
│ Patch 2: [v4-v7]    │ ← Overlaps with Patch 1
├─────────────────────┤
│ Patch 3: [v7-v10]   │ ← Overlaps with Patch 2
├─────────────────────┤
│ Patch 4: [v10-v13]  │ ← Overlaps with Patch 3
└─────────────────────┘
```

### 4. Complete Data Processing Pipeline
*End-to-end flow from multiple data sources to final patches*

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Multiple DBs:   │    │ Filter & Clean  │    │ Combine Sources │
│ • Crypto DB     │ →  │ • Drop NaN      │ →  │ • Merge datasets│
│ • Stock DB      │    │ • Filter indices│    │ • Align indices │
│ • Index DB      │    │ • NORMALIZE     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ↓                       ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Linear Proj +   │ ←  │ Create Patches  │ ←  │ Extract Sequences│
│ Conv Embedding  │    │ • OVERLAPPING   │    │ • seq_len=12    │
│                 │    │ • patch_stride<4│    │ • seq_stride    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 5. Linear Projection & Temporal Embedding
*Shows how patches are transformed into embeddings for model input*

```
Input Patches [4, 4]:                    Linear Projection:
┌─────────────────────┐                  ┌─────────────────────┐
│ Patch 1: [v1-v4]    │ ──────────────→  │ Embed 1: [e1...e64] │
├─────────────────────┤                  ├─────────────────────┤
│ Patch 2: [v4-v7]    │ ──────────────→  │ Embed 2: [e1...e64] │
├─────────────────────┤                  ├─────────────────────┤
│ Patch 3: [v7-v10]   │ ──────────────→  │ Embed 3: [e1...e64] │
├─────────────────────┤                  ├─────────────────────┤
│ Patch 4: [v10-v13]  │ ──────────────→  │ Embed 4: [e1...e64] │
└─────────────────────┘                  └─────────────────────┘
    [4 patches × 4 values]                   [4 patches × 64 dims]

                                ↓ Linear: patch_length → embed_dim

Convolutional Temporal Embedding:
┌─────────────────────────────────────────────────────────────────┐
│                    Conv1D Temporal Processing                   │
│                                                                 │
│ Input:  [4, 64] → Conv1D(kernel=3, stride=1) → [4, 64]        │
│                                                                 │
│ ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                          │
│ │ E1  │───│ E2  │───│ E3  │───│ E4  │  ← Temporal convolution  │
│ │[64] │   │[64] │   │[64] │   │[64] │    captures local        │
│ └─────┘   └─────┘   └─────┘   └─────┘    temporal patterns     │
│    ↓         ↓         ↓         ↓                              │
│ ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                          │
│ │ T1  │   │ T2  │   │ T3  │   │ T4  │  ← Temporally-aware      │
│ │[64] │   │[64] │   │[64] │   │[64] │    embeddings             │
│ └─────┘   └─────┘   └─────┘   └─────┘                          │
└─────────────────────────────────────────────────────────────────┘

Final Model Input: [4, 64] temporally-aware patch embeddings
```

**Key Features:**
- **Multi-source integration**: Combines cryptocurrency, stock, and index data
- **Early normalization**: Applied during filtering phase, not per-batch
- **Overlapping patches**: patch_stride < patch_length creates overlaps
- **Enhanced temporal coverage**: Overlaps capture transitional patterns between patches
- **Unified processing**: All sources normalized together for consistency