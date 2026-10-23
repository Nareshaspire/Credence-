import { ProcessedLoanRecord } from '../types';

export interface HistogramBin {
  binLabel: string;
  min: number;
  max: number;
  paidCount: number;
  defaultCount: number;
  defaultRate: number;
}

export interface CategoricalStat {
  category: string;
  total: number;
  paid: number;
  defaulted: number;
  defaultRate: number; // percentage
}

export interface CorrelationCell {
  var1: string;
  var2: string;
  correlation: number;
}

export interface FeatureSummaryRow {
  featureName: string;
  unit: string;
  overallMean: number;
  paidMean: number;
  defaultMean: number;
  diffPercent: number;
}

export function computeHistogram(
  records: ProcessedLoanRecord[],
  key: keyof ProcessedLoanRecord,
  binCount: number = 8
): HistogramBin[] {
  const values = records.map((r) => Number(r[key]));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const step = (max - min) / binCount;

  const bins: HistogramBin[] = [];
  for (let b = 0; b < binCount; b++) {
    const bMin = min + b * step;
    const bMax = b === binCount - 1 ? max + 0.001 : bMin + step;
    const inBin = records.filter((r) => {
      const v = Number(r[key]);
      return v >= bMin && v < bMax;
    });

    const paid = inBin.filter((r) => r.default === 0).length;
    const def = inBin.filter((r) => r.default === 1).length;
    const rate = inBin.length > 0 ? (def / inBin.length) * 100 : 0;

    bins.push({
      binLabel: `${Math.round(bMin)}-${Math.round(bMax)}`,
      min: bMin,
      max: bMax,
      paidCount: paid,
      defaultCount: def,
      defaultRate: parseFloat(rate.toFixed(1)),
    });
  }

  return bins;
}

export function computeCategoricalStats(
  records: ProcessedLoanRecord[],
  key: 'grade' | 'home_ownership' | 'loan_purpose'
): CategoricalStat[] {
  const groups: Record<string, { total: number; paid: number; def: number }> = {};

  records.forEach((r) => {
    const cat = String(r[key]);
    if (!groups[cat]) {
      groups[cat] = { total: 0, paid: 0, def: 0 };
    }
    groups[cat].total++;
    if (r.default === 1) groups[cat].def++;
    else groups[cat].paid++;
  });

  return Object.entries(groups).map(([category, stats]) => ({
    category,
    total: stats.total,
    paid: stats.paid,
    defaulted: stats.def,
    defaultRate: parseFloat(((stats.def / stats.total) * 100).toFixed(1)),
  }));
}

export function computePearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n === 0) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;

  let num = 0;
  let denX = 0;
  let denY = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }

  const den = Math.sqrt(denX * denY);
  if (den === 0) return 0;
  return parseFloat((num / den).toFixed(2));
}

export function computeCorrelationMatrix(records: ProcessedLoanRecord[]) {
  const variables: { key: keyof ProcessedLoanRecord; label: string }[] = [
    { key: 'fico_score', label: 'FICO' },
    { key: 'dti', label: 'DTI %' },
    { key: 'int_rate', label: 'Int Rate' },
    { key: 'annual_inc', label: 'Income' },
    { key: 'revol_util', label: 'Revol Util' },
    { key: 'delinq_2yrs', label: 'Delinq' },
    { key: 'loan_amnt', label: 'Loan Amt' },
    { key: 'default', label: 'Default' },
  ];

  const matrix: CorrelationCell[] = [];

  variables.forEach((v1) => {
    const x = records.map((r) => Number(r[v1.key]));
    variables.forEach((v2) => {
      const y = records.map((r) => Number(r[v2.key]));
      const corr = computePearsonCorrelation(x, y);
      matrix.push({ var1: v1.label, var2: v2.label, correlation: corr });
    });
  });

  return { labels: variables.map((v) => v.label), matrix };
}

export function computeSummaryTable(records: ProcessedLoanRecord[]): FeatureSummaryRow[] {
  const paid = records.filter((r) => r.default === 0);
  const def = records.filter((r) => r.default === 1);

  const getMean = (arr: ProcessedLoanRecord[], key: keyof ProcessedLoanRecord) =>
    arr.reduce((acc, r) => acc + Number(r[key]), 0) / arr.length;

  const features: { key: keyof ProcessedLoanRecord; name: string; unit: string; decimals?: number }[] = [
    { key: 'fico_score', name: 'FICO Credit Score', unit: 'pts', decimals: 0 },
    { key: 'dti', name: 'Debt-to-Income (DTI)', unit: '%', decimals: 1 },
    { key: 'int_rate', name: 'Interest Rate APR', unit: '%', decimals: 2 },
    { key: 'annual_inc', name: 'Annual Income', unit: '$', decimals: 0 },
    { key: 'loan_amnt', name: 'Loan Amount', unit: '$', decimals: 0 },
    { key: 'revol_util', name: 'Revolving Line Utilization', unit: '%', decimals: 1 },
    { key: 'delinq_2yrs', name: 'Delinquencies in 2 Years', unit: 'count', decimals: 2 },
    { key: 'inq_last_6mths', name: 'Inquiries in 6 Months', unit: 'count', decimals: 2 },
    { key: 'installment_burden', name: 'Installment to Income Burden', unit: '%', decimals: 1 },
  ];

  return features.map((f) => {
    const oMean = getMean(records, f.key);
    const pMean = getMean(paid, f.key);
    const dMean = getMean(def, f.key);
    const diff = pMean !== 0 ? ((dMean - pMean) / pMean) * 100 : 0;

    return {
      featureName: f.name,
      unit: f.unit,
      overallMean: parseFloat(oMean.toFixed(f.decimals ?? 1)),
      paidMean: parseFloat(pMean.toFixed(f.decimals ?? 1)),
      defaultMean: parseFloat(dMean.toFixed(f.decimals ?? 1)),
      diffPercent: parseFloat(diff.toFixed(1)),
    };
  });
}
