import { Grade, HomeOwnership, LoanPurpose, RawLoanRecord, ProcessedLoanRecord } from '../types';

// Reproducible Pseudo-Random Number Generator (Linear Congruential Generator)
function createRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

const rng = createRng(42);

function randRange(min: number, max: number, r = rng): number {
  return min + r() * (max - min);
}

function randInt(min: number, max: number, r = rng): number {
  return Math.floor(randRange(min, max + 1, r));
}

// Box-Muller transform for normal distribution
function randNormal(mean: number, std: number, r = rng): number {
  let u1 = r();
  while (u1 === 0) u1 = r();
  const u2 = r();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + z * std;
}

const LOAN_PURPOSES: LoanPurpose[] = [
  'debt_consolidation',
  'credit_card',
  'home_improvement',
  'small_business',
  'major_purchase',
  'medical',
];

const HOME_OWNERSHIPS: HomeOwnership[] = ['MORTGAGE', 'RENT', 'OWN'];

export function generateLendingClubDataset(count: number = 1200): RawLoanRecord[] {
  const datasetRng = createRng(1337);
  const records: RawLoanRecord[] = [];

  for (let i = 0; i < count; i++) {
    const id = `LN-${10000 + i}`;
    
    // FICO Score distribution: Normal(695, 45) clamped [580, 850]
    const fico = Math.round(
      Math.max(580, Math.min(850, randNormal(695, 45, datasetRng)))
    );

    // Annual Income: Log-normal distribution roughly $24k - $240k
    const logInc = randNormal(11.1, 0.45, datasetRng);
    const annual_inc = Math.round(Math.max(22000, Math.min(260000, Math.exp(logInc))) / 500) * 500;

    // Debt to Income (DTI): centered ~18.5%, range 4 - 44%
    const dti = parseFloat(
      Math.max(3.5, Math.min(44.0, randNormal(18.5, 6.8, datasetRng))).toFixed(1)
    );

    // Revolving utilization: 0 to 98%
    const revol_util = parseFloat(
      Math.max(1.0, Math.min(99.0, randNormal(48.0, 22.0, datasetRng))).toFixed(1)
    );

    // Loan Amount: $1,000 to $40,000
    const rawLoanAmnt = randNormal(14500, 7500, datasetRng);
    const loan_amnt = Math.round(Math.max(1500, Math.min(40000, rawLoanAmnt)) / 250) * 250;

    // Term: 36 vs 60 months (larger loans skew 60)
    const term_months: 36 | 60 = (loan_amnt > 22000 && datasetRng() > 0.45) || datasetRng() > 0.72 ? 60 : 36;

    // Employment length in years: 0 to 10
    const emp_length_years = Math.min(10, Math.max(0, Math.round(randRange(0, 12, datasetRng))));

    // Home ownership: 48% Mortgage, 40% Rent, 12% Own
    const homeRand = datasetRng();
    const home_ownership: HomeOwnership =
      homeRand < 0.48 ? 'MORTGAGE' : homeRand < 0.88 ? 'RENT' : 'OWN';

    // Delinquencies in last 2 years: Poisson-like, mostly 0
    const delinqRand = datasetRng();
    const delinq_2yrs = delinqRand > 0.82 ? (delinqRand > 0.94 ? (delinqRand > 0.98 ? 3 : 2) : 1) : 0;

    // Inquiries in last 6 months:
    const inqRand = datasetRng();
    const inq_last_6mths = inqRand > 0.75 ? (inqRand > 0.92 ? (inqRand > 0.97 ? 3 : 2) : 1) : 0;

    // Total accounts
    const total_acc = Math.max(3, Math.min(48, Math.round(randNormal(22, 9, datasetRng))));

    // Loan purpose
    const pRand = datasetRng();
    const loan_purpose: LoanPurpose =
      pRand < 0.52
        ? 'debt_consolidation'
        : pRand < 0.74
        ? 'credit_card'
        : pRand < 0.84
        ? 'home_improvement'
        : pRand < 0.91
        ? 'small_business'
        : pRand < 0.96
        ? 'major_purchase'
        : 'medical';

    // Compute credit grade and interest rate according to FICO, DTI & delinquency
    let riskIndex =
      (850 - fico) * 0.35 +
      dti * 1.8 +
      revol_util * 0.4 +
      delinq_2yrs * 18 +
      inq_last_6mths * 12 +
      (term_months === 60 ? 12 : 0) -
      (annual_inc > 85000 ? 15 : 0) -
      (home_ownership === 'MORTGAGE' ? 10 : 0);

    let grade: Grade = 'B';
    let subNum = randInt(1, 5, datasetRng);
    let int_rate = 11.5;

    if (riskIndex < 80) {
      grade = 'A';
      int_rate = parseFloat(randRange(5.3, 8.4, datasetRng).toFixed(2));
    } else if (riskIndex < 115) {
      grade = 'B';
      int_rate = parseFloat(randRange(8.5, 12.8, datasetRng).toFixed(2));
    } else if (riskIndex < 155) {
      grade = 'C';
      int_rate = parseFloat(randRange(12.9, 16.5, datasetRng).toFixed(2));
    } else if (riskIndex < 195) {
      grade = 'D';
      int_rate = parseFloat(randRange(16.6, 21.0, datasetRng).toFixed(2));
    } else if (riskIndex < 240) {
      grade = 'E';
      int_rate = parseFloat(randRange(21.1, 25.5, datasetRng).toFixed(2));
    } else {
      grade = 'F';
      int_rate = parseFloat(randRange(25.6, 29.8, datasetRng).toFixed(2));
    }

    const sub_grade = `${grade}${subNum}`;

    // Target calculation: Default likelihood based on real credit factors + logistic link
    // Standard logistic function: P(Default) = 1 / (1 + exp(-logit))
    const incomeDebtRatio = (annual_inc / 12) / (loan_amnt / term_months);
    const logit =
      -4.2 +
      (700 - fico) * 0.015 +
      (dti - 18) * 0.065 +
      (revol_util - 50) * 0.022 +
      delinq_2yrs * 0.45 +
      inq_last_6mths * 0.35 +
      (int_rate - 12) * 0.08 +
      (term_months === 60 ? 0.35 : 0) +
      (home_ownership === 'RENT' ? 0.25 : home_ownership === 'MORTGAGE' ? -0.2 : 0) +
      (loan_purpose === 'small_business' ? 0.45 : loan_purpose === 'medical' ? 0.3 : 0) -
      (incomeDebtRatio > 8 ? 0.4 : 0);

    const defaultProb = 1 / (1 + Math.exp(-logit));
    const isDefault: 0 | 1 = datasetRng() < defaultProb ? 1 : 0;

    records.push({
      id,
      loan_amnt,
      term_months,
      int_rate,
      grade,
      sub_grade,
      annual_inc,
      emp_length_years,
      home_ownership,
      fico_score: fico,
      dti,
      revol_util,
      total_acc,
      delinq_2yrs,
      inq_last_6mths,
      loan_purpose,
      default: isDefault,
    });
  }

  return records;
}

// Data Preprocessing & Feature Engineering
export function preprocessDataset(rawRecords: RawLoanRecord[]): ProcessedLoanRecord[] {
  return rawRecords.map((r) => {
    const monthly_income = Math.max(1, r.annual_inc / 12);
    // Standard monthly loan installment formula
    const monthly_rate = r.int_rate / 100 / 12;
    const n = r.term_months;
    const monthly_installment =
      (r.loan_amnt * (monthly_rate * Math.pow(1 + monthly_rate, n))) /
      (Math.pow(1 + monthly_rate, n) - 1);

    const installment_burden = parseFloat(
      ((monthly_installment / monthly_income) * 100).toFixed(2)
    );

    // Credit stability index: (FICO / 850 * 70) + ((100 - revol_util) * 0.3)
    const credit_stability_index = parseFloat(
      ((r.fico_score / 850) * 70 + ((100 - r.revol_util) * 0.3)).toFixed(1)
    );

    // Risk interaction: DTI * (850 - FICO) / 270
    const risk_interaction = parseFloat(
      ((r.dti * (850 - r.fico_score)) / 270).toFixed(2)
    );

    // Delinquency severity: delinquencies * 1.8 + inquiries * 1.1
    const delinquency_severity = parseFloat(
      (r.delinq_2yrs * 1.8 + r.inq_last_6mths * 1.1).toFixed(2)
    );

    return {
      ...r,
      monthly_income,
      monthly_installment: parseFloat(monthly_installment.toFixed(2)),
      installment_burden,
      credit_stability_index,
      risk_interaction,
      delinquency_severity,
    };
  });
}

// Global cached dataset instance
export const INITIAL_DATASET = generateLendingClubDataset(1200);
export const INITIAL_PROCESSED_DATASET = preprocessDataset(INITIAL_DATASET);
