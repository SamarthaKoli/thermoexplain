import { afterEach, describe, expect, it, vi } from "vitest";

describe("Stitch page navigation", () => {
  afterEach(() => {
    document.body.innerHTML = "";
    window.history.replaceState(null, "", "/");
    vi.resetModules();
  });

  it("maps existing sidebar and overview links to built page entries", async () => {
    document.body.innerHTML = `
      <a href="#" data-path="overview-dashboard">Overview Dashboard</a>
      <a href="#" data-path="cfd-explorer">CFD Explorer &amp; Design</a>
      <a href="#" data-path="xai-lab">Explainable AI Lab</a>
      <a href="#" data-path="thermal-knowledge">Thermal Knowledge</a>
      <a href="#" data-path="evaluation">Evaluation</a>
      <a href="#" data-path="genai-lab">View GenAI Workspace</a>
      <a href="#" data-path="knowledge-assistant">Thermal Assistant</a>
    `;

    await import("./stitch-navigation");

    expect([...document.querySelectorAll("a")].map((link) => link.getAttribute("href"))).toEqual([
      "overview.html",
      "design.html",
      "explainable.html",
      "thermal.html",
      "validation.html",
      "generative.html",
      "thermal.html",
    ]);
  });

  it("leaves unrecognized placeholder links unchanged", async () => {
    document.body.innerHTML = '<a href="#" data-path="unknown">Unknown</a>';

    await import("./stitch-navigation");

    expect(document.querySelector("a")?.getAttribute("href")).toBe("#");
  });

  it("sets a page-specific title for a Stitch entry", async () => {
    window.history.replaceState(null, "", "/overview.html");

    await import("./stitch-navigation");

    expect(document.title).toBe("Overview Dashboard | ThermoExplain AI");
  });
});