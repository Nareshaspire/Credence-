import { ApplicantFormInput, FeatureMap, ProcessedLoanRecord } from '../types';

export interface FeatureDefinition {
  id: string;
  name: string;
  category: 'raw_numeric' | 'engineered' | 'categorical';
  description: string;
}

export const FEATURE_DEFINITIONS: FeatureDefinition[] = [
  { id: 'fico_score', name: 'FICO Score', category: 'raw_numeric', description: 'Credit score range (580-850)' },
  { id: 'dti', name: 'Debt-to-Income (DTI)', category: 'raw_numeric', description: 'Monthly debt payments divided by gross income' },
  { id: 'annual_inc', name: 'Annual Income', category: 'raw_numeric', description: 'Verified applicant self-reported annual income' },
  { id: 'loan_amnt', name: 'Loan Amount', category: 'raw_numeric', description: 'Total requested principal amount' },
  { id: 'int_rate', name: 'Interest Rate', category: 'raw_numeric', description: 'Assigned interest rate APR' },
  { id: 'revol_util', name: 'Revolving Line Utilization', category: 'raw_numeric', description: 'Percentage of available revolving credit used' },
  { id: 'delinq_2yrs', name: 'Delinquencies (2 Yrs)', category: 'raw_numeric', description: 'Past 30+ days past-due credit events in 24m' },
  { id: 'inq_last_6mths', name: 'Recent Inquiries', category: 'raw_numeric', description: 'Hard credit inquiries in past 6 months' },
  { id: 'emp_length_years', name: 'Employment Length', category: 'raw_numeric', description: 'Years at current employment' },
  { id: 'total_acc', name: 'Total Credit Lines', category: 'raw_numeric', description: 'Number of credit lines on credit report' },
  // Engineered
  { id: 'installment_burden', name: 'Payment-to-Income Burden', category: 'engineered', description: 'Monthly payment as percentage of monthly income' },
  { id: 'credit_stability_index', name: 'Credit Stability Index', category: 'engineered', description: 'Composite index of FICO score and low utilization' },
  { id: 'risk_interaction', name: 'DTI × Credit Deficit', category: 'engineered', description: 'Interaction term amplifying high DTI with subprime credit' },
  { id: 'delinquency_severity', name: 'Delinquency Severity Score', category: 'engineered', description: 'Weighted composite of delinquencies and hard inquiries' },
  // Categorical one-hot
  { id: 'term_60', name: 'Term: 60 Months', category: 'categorical', description: 'Binary indicator for 5-year loan duration' },
  { id: 'home_RENT', name: 'Home: Rent', category: 'categorical', description: 'Applicant rents primary residence' },
  { id: 'home_OWN', name: 'Home: Own Free & Clear', category: 'categorical', description: 'Applicant owns home without mortgage' },
  { id: 'purpose_small_biz', name: 'Purpose: Small Business', category: 'categorical', description: 'Loan purpose for small business commercial venture' },
];

export function extractFeatureVector(record: ProcessedLoanRecord): number[] {
  return [
    record.fico_score,
    record.dti,
    record.annual_inc,
    record.loan_amnt,
    record.int_rate,
    record.revol_util,
    record.delinq_2yrs,
    record.inq_last_6mths,
    record.emp_length_years,
    record.total_acc,
    // Engineered features
    record.installment_burden,
    record.credit_stability_index,
    record.risk_interaction,
    record.delinquency_severity,
    // Categorical binary features
    record.term_months === 60 ? 1 : 0,
    record.home_ownership === 'RENT' ? 1 : 0,
    record.home_ownership === 'OWN' ? 1 : 0,
    record.loan_purpose === 'small_business' ? 1 : 0,
  ];
}

export function computeFeatureScaling(trainX: number[][]): FeatureMap {
  const numFeatures = trainX[0].length;
  const map: FeatureMap = {};

  for (let j = 0; j < numFeatures; j++) {
    const colValues = trainX.map((row) => row[j]);
    const min = Math.min(...colValues);
    const max = Math.max(...colValues);
    const mean = colValues.reduce((acc, v) => acc + v, 0) / colValues.length;
    const variance =
      colValues.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) /
      colValues.length;
    const std = Math.sqrt(variance) || 1.0;

    map[j.toString()] = { mean, std, min, max };
  }

  return map;
}

export function transformFeatures(X: number[][], scalingMap: FeatureMap): number[][] {
  return X.map((row) =>
    row.map((val, j) => {
      const stats = scalingMap[j.toString()];
      if (!stats || stats.std === 0) return val;
      return (val - stats.mean) / stats.std;
    })
  );
}

export interface DatasetSplit {
  trainX: number[][];
  trainY: number[];
  testX: number[][];
  testY: number[];
  trainRecords: ProcessedLoanRecord[];
  testRecords: ProcessedLoanRecord[];
  scalingMap: FeatureMap;
}

export function createStratifiedSplit(
  records: ProcessedLoanRecord[],
  testRatio: number = 0.2
): DatasetSplit {
  // Stratify by default label
  const class0 = records.filter((r) => r.default === 0);
  const class1 = records.filter((r) => r.default === 1);

  const splitClass0Index = Math.floor(class0.length * (1 - testRatio));
  const splitClass1Index = Math.floor(class1.length * (1 - testRatio));

  const trainRecords = [...class0.slice(0, splitClass0Index), ...class1.slice(0, splitClass1Index)];
  const testRecords = [...class0.slice(splitClass0Index), ...class1.slice(splitClass1Index)];

  // Shuffle both sets deterministically
  const shuffle = <T>(arr: T[]): T[] => {
    const copy = [...arr];
    let m = copy.length, t, i;
    while (m) {
      i = Math.floor(Math.abs(Math.sin(m * 999)) * m--);
      t = copy[m];
      copy[m] = copy[i];
      copy[i] = t;
    }
    return copy;
  };

  const shuffledTrain = shuffle(trainRecords);
  const shuffledTest = shuffle(testRecords);

  const rawTrainX = shuffledTrain.map(extractFeatureVector);
  const trainY = shuffledTrain.map((r) => r.default as number);

  const rawTestX = shuffledTest.map(extractFeatureVector);
  const testY = shuffledTest.map((r) => r.default as number);

  const scalingMap = computeFeatureScaling(rawTrainX);
  const trainX = transformFeatures(rawTrainX, scalingMap);
  const testX = transformFeatures(rawTestX, scalingMap);

  return {
    trainX,
    trainY,
    testX,
    testY,
    trainRecords: shuffledTrain,
    testRecords: shuffledTest,
    scalingMap,
  };
}

export function transformApplicantInput(
  input: ApplicantFormInput,
  scalingMap: FeatureMap
): { rawVector: number[]; normalizedVector: number[] } {
  // Compute monthly installment
  // Estimate interest rate based on FICO and DTI
  let estRate = 12.0;
  if (input.fico_score >= 760) estRate = 6.8;
  else if (input.fico_score >= 720) estRate = 9.2;
  else if (input.fico_score >= 680) estRate = 13.5;
  else if (input.fico_score >= 640) estRate = 18.2;
  else estRate = 24.5;

  const monthly_income = Math.max(1, input.annual_inc / 12);
  const monthly_rate = estRate / 100 / 12;
  const n = input.term_months;
  const monthly_installment =
    (input.loan_amnt * (monthly_rate * Math.pow(1 + monthly_rate, n))) /
    (Math.pow(1 + monthly_rate, n) - 1);

  const installment_burden = (monthly_installment / monthly_income) * 100;
  const credit_stability_index = (input.fico_score / 850) * 70 + ((100 - input.revol_util) * 0.3);
  const risk_interaction = (input.dti * (850 - input.fico_score)) / 270;
  const delinquency_severity = input.delinq_2yrs * 1.8 + input.inq_last_6mths * 1.1;

  const rawVector = [
    input.fico_score,
    input.dti,
    input.annual_inc,
    input.loan_amnt,
    estRate,
    input.revol_util,
    input.delinq_2yrs,
    input.inq_last_6mths,
    input.emp_length_years,
    input.total_acc,
    installment_burden,
    credit_stability_index,
    risk_interaction,
    delinquency_severity,
    input.term_months === 60 ? 1 : 0,
    input.home_ownership === 'RENT' ? 1 : 0,
    input.home_ownership === 'OWN' ? 1 : 0,
    input.loan_purpose === 'small_business' ? 1 : 0,
  ];

  const normalizedVector = rawVector.map((val, j) => {
    const stats = scalingMap[j.toString()];
    if (!stats || stats.std === 0) return val;
    return (val - stats.mean) / stats.std;
  });

  return { rawVector, normalizedVector };
}
