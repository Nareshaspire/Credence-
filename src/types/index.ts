export type Grade = 'A' | 'B' | 'C' | 'D' | 'E' | 'F';
export type HomeOwnership = 'MORTGAGE' | 'RENT' | 'OWN';
export type LoanPurpose =
  | 'debt_consolidation'
  | 'credit_card'
  | 'home_improvement'
  | 'small_business'
  | 'major_purchase'
  | 'medical';

export interface RawLoanRecord {
  id: string;
  loan_amnt: number;
  term_months: 36 | 60;
  int_rate: number; // e.g. 11.5%
  grade: Grade;
  sub_grade: string;
  annual_inc: number;
  emp_length_years: number;
  home_ownership: HomeOwnership;
  fico_score: number;
  dti: number; // Debt to income %
  revol_util: number; // Revolving line utilization %
  total_acc: number;
  delinq_2yrs: number;
  inq_last_6mths: number;
  loan_purpose: LoanPurpose;
  default: 0 | 1; // Target: 1 = defaulted, 0 = paid in full
}

export interface ProcessedLoanRecord extends RawLoanRecord {
  // Engineered features
  monthly_income: number;
  monthly_installment: number;
  installment_burden: number; // installment / monthly_income
  credit_stability_index: number; // composite FICO & revol_util
  risk_interaction: number; // DTI * (850 - FICO) / 270
  delinquency_severity: number;
}

export interface FeatureNormalization {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export type FeatureMap = Record<string, FeatureNormalization>;

export type ModelAlgorithm =
  | 'logistic_regression'
  | 'random_forest'
  | 'gradient_boosting'
  | 'decision_tree'
  | 'k_nearest_neighbors';

export interface ModelHyperparameters {
  learningRate?: number;
  maxDepth?: number;
  nEstimators?: number;
  regularizationC?: number;
  minSamplesSplit?: number;
  kNeighbors?: number;
  decisionThreshold: number;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface CurvePoint {
  x: number; // FPR or Recall
  y: number; // TPR or Precision
  threshold: number;
}

export interface FeatureImportance {
  feature: string;
  displayName: string;
  importance: number; // 0 to 1
  impactDirection: 'positive' | 'negative' | 'neutral'; // on default risk
}

export interface ModelMetrics {
  algorithm: ModelAlgorithm;
  algorithmName: string;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1Score: number;
  rocAuc: number;
  prAuc: number;
  logLoss: number;
  trainingTimeMs: number;
  confusionMatrix: ConfusionMatrix;
  rocCurve: CurvePoint[];
  prCurve: CurvePoint[];
  featureImportances: FeatureImportance[];
  hyperparameters: ModelHyperparameters;
}

export interface PredictionResult {
  defaultProbability: number; // 0 to 1
  predictedDefault: boolean;
  riskTier: 'AAA' | 'AA' | 'A' | 'BBB' | 'BB' | 'B' | 'CCC' | 'D';
  recommendation: 'Approved' | 'Conditional Review' | 'High Risk Declined';
  recommendedRate: number; // suggested APR %
  expectedLoss: number;
  featureContributions: {
    feature: string;
    label: string;
    contribution: number; // percentage delta in risk
    isProtective: boolean; // lowers risk vs increases risk
    valueDisplay: string;
  }[];
}

export interface ApplicantFormInput {
  loan_amnt: number;
  term_months: 36 | 60;
  annual_inc: number;
  emp_length_years: number;
  home_ownership: HomeOwnership;
  fico_score: number;
  dti: number;
  revol_util: number;
  total_acc: number;
  delinq_2yrs: number;
  inq_last_6mths: number;
  loan_purpose: LoanPurpose;
}
