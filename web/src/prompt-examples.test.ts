// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest"

import {
  DEFAULT_PROMPT,
  matchesPromptExample,
  PROMPT_EXAMPLE_CATEGORIES,
  PROMPT_EXAMPLE_GROUPS,
  PROMPT_EXAMPLES,
} from "./prompt-examples"

describe("prompt examples", () => {
  it("contains exactly 100 polished browser-demo prompts", () => {
    expect(PROMPT_EXAMPLES).toHaveLength(100)
    expect(PROMPT_EXAMPLES[0].prompt).toBe(DEFAULT_PROMPT)
    expect(DEFAULT_PROMPT).toBe(
      "A person walks forward, then waves with their right hand."
    )

    const normalizedLabels = PROMPT_EXAMPLES.map(({ label }) =>
      label.toLocaleLowerCase("en-US")
    )
    const normalizedPrompts = PROMPT_EXAMPLES.map(({ prompt }) =>
      prompt.toLocaleLowerCase("en-US")
    )
    expect(new Set(normalizedLabels)).toHaveLength(PROMPT_EXAMPLES.length)
    expect(new Set(normalizedPrompts)).toHaveLength(PROMPT_EXAMPLES.length)

    for (const example of PROMPT_EXAMPLES) {
      expect(example.label).toBe(example.label.trim())
      expect(example.prompt).toBe(example.prompt.trim())
      expect(example.label.length).toBeGreaterThan(0)
      expect(example.prompt.length).toBeGreaterThan(0)
      expect(example.prompt.length).toBeLessThanOrEqual(280)
      expect(example.label).toMatch(/^[\x20-\x7e]+$/)
      expect(example.prompt).toMatch(/^[\x20-\x7e]+$/)
      expect(example.prompt).toMatch(/[.!?]$/)
      expect(PROMPT_EXAMPLE_CATEGORIES).toContain(example.category)
    }
  })

  it("keeps ten complete categories in a stable display order", () => {
    expect(PROMPT_EXAMPLE_CATEGORIES).toHaveLength(10)
    expect(PROMPT_EXAMPLE_GROUPS.map(({ value }) => value)).toEqual(
      PROMPT_EXAMPLE_CATEGORIES
    )
    for (const group of PROMPT_EXAMPLE_GROUPS) {
      expect(group.items).toHaveLength(10)
      expect(
        group.items.every((example) => example.category === group.value)
      ).toBe(true)
    }
    expect(PROMPT_EXAMPLE_GROUPS.flatMap(({ items }) => items)).toEqual(
      PROMPT_EXAMPLES
    )
  })

  it("searches labels, prompt text, and categories without case sensitivity", () => {
    const confidentWalk = PROMPT_EXAMPLES.find(
      ({ label }) => label === "Confident walk"
    )
    const pullRope = PROMPT_EXAMPLES.find(
      ({ label }) => label === "Pull rope"
    )
    if (!confidentWalk || !pullRope) {
      throw new Error("Required prompt examples are missing")
    }

    expect(matchesPromptExample(confidentWalk, "")).toBe(true)
    expect(matchesPromptExample(confidentWalk, "CONFIDENT")).toBe(true)
    expect(matchesPromptExample(confidentWalk, "relaxed swings")).toBe(true)
    expect(matchesPromptExample(pullRope, "pantomime heavy")).toBe(true)
    expect(matchesPromptExample(pullRope, "  pull   HEAVY  ")).toBe(true)
    expect(matchesPromptExample(pullRope, "pirouette")).toBe(false)
  })
})
