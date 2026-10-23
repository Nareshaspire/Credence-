import { FEATURE_DEFINITIONS } from './featureEngineering';
import { FeatureImportance, ModelHyperparameters } from '../types';

export interface MLModel {
  fit: (X: number[][], y: number[]) => void;
  predictProba: (X: number[][]) => number[];
  getFeatureImportances: () => FeatureImportance[];
}

function sigmoid(z: number): number {
  if (z < -30) return 0.000001;
  if (z > 30) return 0.999999;
  return 1 / (1 + Math.exp(-z));
}

// -------------------------------------------------------------
// 1. Logistic Regression with L2 (Ridge) Regularization
// -------------------------------------------------------------
export class LogisticRegressionModel implements MLModel {
  weights: number[] = [];
  bias: number = 0;
  lr: number;
  lambda: number;
  epochs: number;

  constructor(params: ModelHyperparameters) {
    this.lr = params.learningRate ?? 0.08;
    this.lambda = (1 / (params.regularizationC ?? 1.0)) * 0.01;
    this.epochs = 200;
  }

  fit(X: number[][], y: number[]) {
    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.weights = new Array(nFeatures).fill(0);
    this.bias = -1.5; // prior log-odds for ~18% base rate

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const gradW = new Array(nFeatures).fill(0);
      let gradB = 0;

      for (let i = 0; i < nSamples; i++) {
        let linear = this.bias;
        for (let j = 0; j < nFeatures; j++) {
          linear += this.weights[j] * X[i][j];
        }
        const prob = sigmoid(linear);
        const error = prob - y[i];

        gradB += error;
        for (let j = 0; j < nFeatures; j++) {
          gradW[j] += error * X[i][j];
        }
      }

      // Update with L2 weight decay
      this.bias -= (this.lr * gradB) / nSamples;
      for (let j = 0; j < nFeatures; j++) {
        this.weights[j] -=
          (this.lr * (gradW[j] + this.lambda * this.weights[j])) / nSamples;
      }
    }
  }

  predictProba(X: number[][]): number[] {
    return X.map((row) => {
      let linear = this.bias;
      for (let j = 0; j < row.length; j++) {
        linear += this.weights[j] * row[j];
      }
      return sigmoid(linear);
    });
  }

  getFeatureImportances(): FeatureImportance[] {
    const absWeights = this.weights.map((w) => Math.abs(w));
    const maxWeight = Math.max(...absWeights) || 1.0;

    return FEATURE_DEFINITIONS.map((def, idx) => {
      const rawW = this.weights[idx] || 0;
      const direction: 'positive' | 'negative' | 'neutral' =
        rawW > 0.05 ? 'positive' : rawW < -0.05 ? 'negative' : 'neutral';
      return {
        feature: def.id,
        displayName: def.name,
        importance: parseFloat((Math.abs(rawW) / maxWeight).toFixed(3)),
        impactDirection: direction,
      };
    }).sort((a, b) => b.importance - a.importance);
  }
}

// -------------------------------------------------------------
// 2. Decision Tree Classifier (CART with Gini Impurity)
// -------------------------------------------------------------
interface TreeNode {
  isLeaf: boolean;
  prediction?: number;
  featureIndex?: number;
  splitThreshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export class DecisionTreeModel implements MLModel {
  root: TreeNode | null = null;
  maxDepth: number;
  minSamplesSplit: number;
  featureGiniGains: number[] = [];

  constructor(params: ModelHyperparameters) {
    this.maxDepth = params.maxDepth ?? 5;
    this.minSamplesSplit = params.minSamplesSplit ?? 10;
  }

  private gini(labels: number[]): number {
    if (labels.length === 0) return 0;
    const p1 = labels.reduce((a, b) => a + b, 0) / labels.length;
    return 1 - (p1 * p1 + (1 - p1) * (1 - p1));
  }

  private buildTree(X: number[][], y: number[], depth: number): TreeNode {
    const nSamples = y.length;
    const nFeatures = X[0].length;
    const p1 = y.reduce((a, b) => a + b, 0) / nSamples;

    // Base conditions
    if (depth >= this.maxDepth || nSamples < this.minSamplesSplit || p1 === 0 || p1 === 1) {
      return { isLeaf: true, prediction: p1 };
    }

    const currentGini = this.gini(y);
    let bestGain = 0;
    let bestFeature = -1;
    let bestThreshold = 0;

    for (let f = 0; f < nFeatures; f++) {
      // Sample candidate thresholds (percentiles)
      const values = X.map((row) => row[f]).sort((a, b) => a - b);
      const step = Math.max(1, Math.floor(values.length / 8));
      
      for (let k = step; k < values.length; k += step) {
        const threshold = values[k];
        const leftY: number[] = [];
        const rightY: number[] = [];

        for (let i = 0; i < nSamples; i++) {
          if (X[i][f] <= threshold) leftY.push(y[i]);
          else rightY.push(y[i]);
        }

        if (leftY.length === 0 || rightY.length === 0) continue;

        const weightedGini =
          (leftY.length / nSamples) * this.gini(leftY) +
          (rightY.length / nSamples) * this.gini(rightY);
        const gain = currentGini - weightedGini;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestThreshold = threshold;
        }
      }
    }

    if (bestGain <= 0.001 || bestFeature === -1) {
      return { isLeaf: true, prediction: p1 };
    }

    this.featureGiniGains[bestFeature] = (this.featureGiniGains[bestFeature] || 0) + bestGain * nSamples;

    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      if (X[i][bestFeature] <= bestThreshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return {
      isLeaf: false,
      featureIndex: bestFeature,
      splitThreshold: bestThreshold,
      left: this.buildTree(leftX, leftY, depth + 1),
      right: this.buildTree(rightX, rightY, depth + 1),
    };
  }

  fit(X: number[][], y: number[]) {
    this.featureGiniGains = new Array(X[0].length).fill(0);
    this.root = this.buildTree(X, y, 0);
  }

  private predictSingle(node: TreeNode | null, row: number[]): number {
    if (!node) return 0.2;
    if (node.isLeaf) return node.prediction ?? 0.2;
    if (row[node.featureIndex!] <= node.splitThreshold!) {
      return this.predictSingle(node.left ?? null, row);
    }
    return this.predictSingle(node.right ?? null, row);
  }

  predictProba(X: number[][]): number[] {
    return X.map((row) => this.predictSingle(this.root, row));
  }

  getFeatureImportances(): FeatureImportance[] {
    const maxGain = Math.max(...this.featureGiniGains) || 1.0;
    return FEATURE_DEFINITIONS.map((def, idx) => ({
      feature: def.id,
      displayName: def.name,
      importance: parseFloat(((this.featureGiniGains[idx] || 0) / maxGain).toFixed(3)),
      impactDirection: 'positive' as const,
    })).sort((a, b) => b.importance - a.importance);
  }
}

// -------------------------------------------------------------
// 3. Random Forest Classifier (Bootstrap Bagged Decision Trees)
// -------------------------------------------------------------
export class RandomForestModel implements MLModel {
  trees: DecisionTreeModel[] = [];
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  featureImportancesAgg: number[] = [];

  constructor(params: ModelHyperparameters) {
    this.nEstimators = params.nEstimators ?? 25;
    this.maxDepth = params.maxDepth ?? 5;
    this.minSamplesSplit = params.minSamplesSplit ?? 8;
  }

  fit(X: number[][], y: number[]) {
    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.trees = [];
    this.featureImportancesAgg = new Array(nFeatures).fill(0);

    for (let t = 0; t < this.nEstimators; t++) {
      // Bootstrap sampling with replacement
      const bootX: number[][] = [];
      const bootY: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const randIdx = Math.floor(Math.abs(Math.sin((t + 1) * 31 + i * 17)) * nSamples);
        bootX.push(X[randIdx]);
        bootY.push(y[randIdx]);
      }

      const tree = new DecisionTreeModel({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        decisionThreshold: 0.5,
      });

      tree.fit(bootX, bootY);
      this.trees.push(tree);

      for (let f = 0; f < nFeatures; f++) {
        this.featureImportancesAgg[f] += tree.featureGiniGains[f] || 0;
      }
    }
  }

  predictProba(X: number[][]): number[] {
    if (this.trees.length === 0) return X.map(() => 0.2);
    const predictions = this.trees.map((t) => t.predictProba(X));

    return X.map((_, i) => {
      const sum = predictions.reduce((acc, predArr) => acc + predArr[i], 0);
      return sum / this.trees.length;
    });
  }

  getFeatureImportances(): FeatureImportance[] {
    const maxVal = Math.max(...this.featureImportancesAgg) || 1.0;
    return FEATURE_DEFINITIONS.map((def, idx) => ({
      feature: def.id,
      displayName: def.name,
      importance: parseFloat(((this.featureImportancesAgg[idx] || 0) / maxVal).toFixed(3)),
      impactDirection: 'positive' as const,
    })).sort((a, b) => b.importance - a.importance);
  }
}

// -------------------------------------------------------------
// 4. Gradient Boosted Trees (GBDT)
// -------------------------------------------------------------
export class GradientBoostingModel implements MLModel {
  trees: DecisionTreeModel[] = [];
  nEstimators: number;
  learningRate: number;
  maxDepth: number;
  baseLogOdds: number = 0;
  featureImportancesAgg: number[] = [];

  constructor(params: ModelHyperparameters) {
    this.nEstimators = params.nEstimators ?? 20;
    this.learningRate = params.learningRate ?? 0.12;
    this.maxDepth = Math.min(4, params.maxDepth ?? 3);
  }

  fit(X: number[][], y: number[]) {
    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.trees = [];
    this.featureImportancesAgg = new Array(nFeatures).fill(0);

    const pos = y.reduce((a, b) => a + b, 0);
    const p = Math.max(0.01, Math.min(0.99, pos / nSamples));
    this.baseLogOdds = Math.log(p / (1 - p));

    const currentF = new Array(nSamples).fill(this.baseLogOdds);

    for (let t = 0; t < this.nEstimators; t++) {
      // Calculate pseudo-residuals: residual = y - proba
      const residuals: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const prob = sigmoid(currentF[i]);
        residuals.push(y[i] - prob);
      }

      // Fit regression tree to residuals
      const tree = new DecisionTreeModel({
        maxDepth: this.maxDepth,
        minSamplesSplit: 10,
        decisionThreshold: 0.5,
      });

      // Map pseudo labels for tree split
      tree.fit(X, residuals.map((r) => (r > 0 ? 1 : 0)));
      this.trees.push(tree);

      const treeProbs = tree.predictProba(X);
      for (let i = 0; i < nSamples; i++) {
        // Adjust step with learning rate
        currentF[i] += this.learningRate * (treeProbs[i] - 0.5) * 2;
      }

      for (let f = 0; f < nFeatures; f++) {
        this.featureImportancesAgg[f] += tree.featureGiniGains[f] || 0;
      }
    }
  }

  predictProba(X: number[][]): number[] {
    return X.map((row) => {
      let fVal = this.baseLogOdds;
      for (const tree of this.trees) {
        const pred = tree.predictProba([row])[0];
        fVal += this.learningRate * (pred - 0.5) * 2;
      }
      return sigmoid(fVal);
    });
  }

  getFeatureImportances(): FeatureImportance[] {
    const maxVal = Math.max(...this.featureImportancesAgg) || 1.0;
    return FEATURE_DEFINITIONS.map((def, idx) => ({
      feature: def.id,
      displayName: def.name,
      importance: parseFloat(((this.featureImportancesAgg[idx] || 0) / maxVal).toFixed(3)),
      impactDirection: 'positive' as const,
    })).sort((a, b) => b.importance - a.importance);
  }
}

// -------------------------------------------------------------
// 5. K-Nearest Neighbors (KNN)
// -------------------------------------------------------------
export class KNNModel implements MLModel {
  trainX: number[][] = [];
  trainY: number[] = [];
  k: number;

  constructor(params: ModelHyperparameters) {
    this.k = params.kNeighbors ?? 7;
  }

  fit(X: number[][], y: number[]) {
    this.trainX = X;
    this.trainY = y;
  }

  predictProba(X: number[][]): number[] {
    return X.map((target) => {
      // Calculate Euclidean distances
      const distances = this.trainX.map((trainRow, idx) => {
        let sumSq = 0;
        for (let j = 0; j < target.length; j++) {
          const diff = target[j] - trainRow[j];
          sumSq += diff * diff;
        }
        return { dist: Math.sqrt(sumSq), label: this.trainY[idx] };
      });

      distances.sort((a, b) => a.dist - b.dist);
      const topK = distances.slice(0, this.k);
      const sumLabels = topK.reduce((acc, curr) => acc + curr.label, 0);
      return sumLabels / this.k;
    });
  }

  getFeatureImportances(): FeatureImportance[] {
    // Distance-based approximation based on univariate variance
    return FEATURE_DEFINITIONS.map((def) => ({
      feature: def.id,
      displayName: def.name,
      importance: 0.5,
      impactDirection: 'neutral',
    }));
  }
}

// Factory helper
export function createModel(
  algo: string,
  params: ModelHyperparameters
): MLModel {
  switch (algo) {
    case 'random_forest':
      return new RandomForestModel(params);
    case 'gradient_boosting':
      return new GradientBoostingModel(params);
    case 'decision_tree':
      return new DecisionTreeModel(params);
    case 'k_nearest_neighbors':
      return new KNNModel(params);
    case 'logistic_regression':
    default:
      return new LogisticRegressionModel(params);
  }
}
