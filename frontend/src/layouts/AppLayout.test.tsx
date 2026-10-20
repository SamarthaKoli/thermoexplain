import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AppLayout, type PageId } from "./AppLayout";

describe("AppLayout navigation", () => {
  it("exposes all six sections and marks the selected destination", () => {
    const onNavigate = vi.fn<(page: PageId) => void>();
    render(
      <AppLayout activePage="overview" onNavigate={onNavigate}>
        <h1>Overview content</h1>
      </AppLayout>,
    );

    const navigation = screen.getByRole("navigation", { name: "Main navigation" });
    expect(screen.getAllByRole("button", { name: /Overview|CFD Explorer|Explainable AI Lab|Generative AI Lab|Knowledge Assistant|Evaluation & Validation/ })).toHaveLength(6);
    expect(screen.getByRole("button", { name: /Overview/ })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("main")).toHaveTextContent("Overview content");

    fireEvent.click(screen.getByRole("button", { name: /Evaluation & Validation/ }));
    expect(onNavigate).toHaveBeenCalledWith("evaluation");
    expect(navigation).toBeInTheDocument();
  });
});
