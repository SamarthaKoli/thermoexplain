const pageByPath: Record<string, string> = {
  "overview-dashboard": "overview.html",
  "cfd-explorer": "design.html",
  "cfd-explorer-design-analysis": "design.html",
  "cfd-explorer-and-design-analysis": "design.html",
  "explainable-ai": "explainable.html",
  "explainable-ai-lab": "explainable.html",
  "xai-lab": "explainable.html",
  "generative-ai": "generative.html",
  "generative-ai-lab": "generative.html",
  "genai-lab": "generative.html",
  "thermal-knowledge": "thermal.html",
  "thermal-knowledge-assistant": "thermal.html",
  "knowledge-assistant": "thermal.html",
  evaluation: "validation.html",
  "evaluation-validation": "validation.html",
  "evaluation-and-validation": "validation.html",
  "thermoexplain-ai-overview-dashboard": "overview.html",
  "thermoexplain-ai-cfd-explorer-design-analysis": "design.html",
  "thermoexplain-ai-explainable-ai-lab": "explainable.html",
  "thermoexplain-ai-generative-ai-lab": "generative.html",
  "thermoexplain-ai-thermal-knowledge-assistant": "thermal.html",
  "thermoexplain-ai-evaluation-validation": "validation.html",
};

const pageByLabel = [
  [/^overview dashboard$/i, "overview.html"],
  [/^cfd explorer(?: & design analysis)?$/i, "design.html"],
  [/^explainable ai lab$/i, "explainable.html"],
  [/^generative ai lab$/i, "generative.html"],
  [/^thermal knowledge assistant$/i, "thermal.html"],
  [/^evaluation & validation$/i, "validation.html"],
] as const;

const titleByPath: Record<string, string> = {
  "/overview.html": "Overview Dashboard | ThermoExplain AI",
  "/design.html": "CFD Explorer & Design Analysis | ThermoExplain AI",
  "/explainable.html": "Explainable AI Lab | ThermoExplain AI",
  "/generative.html": "Generative AI Lab | ThermoExplain AI",
  "/thermal.html": "Thermal Knowledge Assistant | ThermoExplain AI",
  "/validation.html": "Evaluation & Validation | ThermoExplain AI",
};

document.title = titleByPath[window.location.pathname.toLowerCase()] ?? "ThermoExplain AI";

for (const link of document.querySelectorAll<HTMLAnchorElement>('a[href="#"]')) {
  const normalizedPath = link.dataset.path?.toLowerCase();
  const label = link.textContent?.replace(/\s+/g, " ").trim() ?? "";
  const destination = (normalizedPath && pageByPath[normalizedPath])
    ?? pageByLabel.find(([pattern]) => pattern.test(label))?.[1];

  if (destination) link.href = destination;
}

export {};