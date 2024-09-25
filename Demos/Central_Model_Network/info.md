## Base Data [Location 2 + Location 3]

| Total Fraud | Total Non-Fraud | Total Transactions |
|-------------|-----------------|--------------------|
| 301         | 17629           | 17930              |

Percent of Fraudulent Transactions: 1.68%

# Case 1: Simple generation of data

## Synthetic Data [Location 1]

| Total Fraud | Total Non-Fraud | Total Transactions |
|-------------|-----------------|--------------------|
| 4           | 13444           | 13448              |

Percent of Fraudulent Transactions: 0.03%

## Blend Data [Synthetic + Base Data]

| Total Fraud | Total Non-Fraud | Total Transactions |
|-------------|-----------------|--------------------|
| 305         | 31073           | 31378              |

Percent of Fraudulent Transactions: 0.97%

## Classification Results

| Model      | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------|----------|-----------|--------|----------|---------|
| Base Data  | .994     | .850      | .472   | .607     | .736    |
| Blend Data | .992     | .875      | .292   | .438     | .646    |

# Case 2: Generation of data with amplify to boost fraud

## Synthetic Data [Location 1]

| Total Fraud | Total Non-Fraud | Total Transactions |
|-------------|-----------------|--------------------|
| 1054        | 6332            | 7386               |

Percent of Fraudulent Transactions: 14.27%

## Blended Data [Synthetic + Base Data]

| Total Fraud | Total Non-Fraud | Total Transactions |
|-------------|-----------------|--------------------|
| 1335        | 23961           | 25316              |

Percent of Fraudulent Transactions: 5.35%

## Classification Results

| Model      | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------|----------|-----------|--------|----------|---------|
| Base Data  | .994     | .850      | .472   | .607     | .736    |
| Blend Data | .995     | .927      | .528   | .673     | .764    |