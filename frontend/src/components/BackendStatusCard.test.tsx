import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { BackendStatusCard } from "./BackendStatusCard";

describe("BackendStatusCard", () => {
  it("shows a successful backend connection", () => {
    render(
      <BackendStatusCard
        state="connected"
        message="Backend is responding."
        onRetry={vi.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Backend is responding.",
    );
  });

  it("offers retry when the backend is unavailable", () => {
    render(
      <BackendStatusCard
        state="unavailable"
        message="Could not reach the backend."
        onRetry={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});
