import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiRequestError,
  getAQI,
  getAvailableDates,
  getHealth,
  getPM25,
} from "./api";

describe("api client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("returns health data on success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ status: "ok", service: "pm25-mapping-api" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const result = await getHealth();
    expect(result.status).toBe("ok");
    expect(result.service).toBe("pm25-mapping-api");
  });

  it("returns parsed available dates", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ dates: ["2025-01-01"] }), { status: 200 }),
    );
    const result = await getAvailableDates();
    expect(result.dates).toEqual(["2025-01-01"]);
  });

  it("throws ApiRequestError with server detail on HTTP 404", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Requested date output is not available." }), {
        status: 404,
      }),
    );
    await expect(getPM25("2099-01-01", 28.6, 77.2)).rejects.toThrow(
      ApiRequestError,
    );
    await expect(getPM25("2099-01-01", 28.6, 77.2)).rejects.toMatchObject({
      status: 404,
    });
  });

  it("maps network failure to a Backend unavailable error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("failed to fetch"));
    await expect(getAQI("2025-01-01", 28.6, 77.2)).rejects.toThrow(
      "Backend unavailable",
    );
  });
});
