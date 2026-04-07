"""Tests for shared textbook backstop recommendations."""

from __future__ import annotations

import unittest

try:
    from functorflow_v3.textbook_backstop import recommend_textbook_backstop, render_textbook_backstop_html
except ModuleNotFoundError:
    from ..functorflow_v3.textbook_backstop import recommend_textbook_backstop, render_textbook_backstop_html


class TextbookBackstopTests(unittest.TestCase):
    def test_company_similarity_query_recommends_diffusion_and_geometry_sections(self) -> None:
        backstop = recommend_textbook_backstop(
            "How similar is Adobe to Nike?",
            route_name="company_similarity",
        )

        section_titles = {section.title for section in backstop.sections}
        self.assertIn("Temporal Diffusion over Causal Trajectories", section_titles)
        self.assertIn("Manifold Learning with Geometric Transformers", section_titles)

    def test_democritus_query_recommends_csql_and_causality_sections(self) -> None:
        backstop = recommend_textbook_backstop(
            "Give me 5 studies of global warming and synthesize their joint claims",
            route_name="democritus",
        )

        section_titles = {section.title for section in backstop.sections}
        self.assertIn("CSQL: Mapping Documents into Topos Causal Model Databases", section_titles)
        self.assertIn("Causality from Language", section_titles)

    def test_product_feedback_query_recommends_feedback_relevant_sections(self) -> None:
        backstop = recommend_textbook_backstop(
            "How easy is it to drive a Tesla Model 3?",
            route_name="product_feedback",
        )

        section_titles = {section.title for section in backstop.sections}
        self.assertIn("Causality from Language", section_titles)
        self.assertIn("Topos Causal Models", section_titles)

    def test_basket_rocket_query_recommends_agentic_and_code_sections(self) -> None:
        backstop = recommend_textbook_backstop(
            "Find me 10 recent AMD 10-K filings",
            route_name="basket_rocket_sec",
        )

        section_titles = {section.title for section in backstop.sections}
        self.assertIn("Building Agentic Systems using Kan Extension Transformers", section_titles)
        self.assertIn("Code Companion", section_titles)

    def test_rendered_html_mentions_textbook_source_and_pages(self) -> None:
        backstop = recommend_textbook_backstop(
            "How similar is Adobe to Nike?",
            route_name="company_similarity",
        )

        html = render_textbook_backstop_html(backstop)

        self.assertIn("Read This in the Book", html)
        self.assertIn("catagi.pdf", html)
        self.assertIn("page", html)


if __name__ == "__main__":
    unittest.main()
