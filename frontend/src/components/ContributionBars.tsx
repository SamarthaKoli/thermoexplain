interface ContributionBarItem {
  feature: string;
  contribution: number;
  meanAbsoluteShap?: number;
  meanShap?: number;
}

interface ContributionBarsProps {
  items: ContributionBarItem[];
  global?: boolean;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat(undefined, {
    maximumSignificantDigits: 6,
  }).format(value);
}

export function ContributionBars({
  items,
  global = false,
}: ContributionBarsProps) {
  const maximum = Math.max(
    ...items.map((item) =>
      global ? item.meanAbsoluteShap ?? 0 : Math.abs(item.contribution),
    ),
    0,
  );
  if (items.length === 0) {
    return <p className="muted">No computed explanation values are available.</p>;
  }
  return (
    <ul className="contribution-list">
      {items.map((item) => {
        const value = global ? item.meanAbsoluteShap ?? 0 : item.contribution;
        const width = maximum > 0 ? (Math.abs(value) / maximum) * 100 : 0;
        return (
          <li className="contribution-list__item" key={item.feature}>
            <div className="contribution-list__label">
              <strong>{item.feature}</strong>
              <span>
                {global
                  ? `Mean |SHAP| ${formatNumber(value)} · signed mean ${formatNumber(item.meanShap ?? 0)}`
                  : `${value >= 0 ? "+" : ""}${formatNumber(value)}`}
              </span>
            </div>
            <div className="contribution-track" aria-hidden="true">
              <span
                className={`contribution-track__value${!global && value < 0 ? " contribution-track__value--negative" : ""}`}
                style={{ width: `${width}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
