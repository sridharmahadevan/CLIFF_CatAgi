"""Tests for the product-feedback agentic scaffold."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from functorflow_v3 import (
        ProductFeedbackAgenticConfig,
        ProductFeedbackAgenticRunner,
        build_product_feedback_agentic_workflow,
    )
except ModuleNotFoundError:
    from ..functorflow_v3 import (
        ProductFeedbackAgenticConfig,
        ProductFeedbackAgenticRunner,
        build_product_feedback_agentic_workflow,
    )


class ProductFeedbackAgenticTests(unittest.TestCase):
    def test_product_usage_family_detects_vehicle_queries(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        family = module._product_usage_family(
            "Tesla Model 3",
            "Tesla",
            ["The steering is easy, charging is convenient, and the car is comfortable on long drives."],
        )

        self.assertEqual(family, "vehicle")

    def test_product_usage_family_detects_food_queries(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        family = module._product_usage_family(
            "Amedei Porcelana Chocolate Bars",
            "Amedei",
            ["The chocolate tastes rich, the flavor is nuanced, and we shared it after dessert."],
        )

        self.assertEqual(family, "food")

    def test_counterfactuals_infer_running_shoe_repairs_from_text_cues(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "run1",
                    "title": "Fast but not secure",
                    "text": (
                        "Navigation boilerplate and collection filters. The shoe felt harsh after long runs "
                        "and the heel slips when I pick up the pace. More unrelated page text follows."
                    ),
                    "rating": 3,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["style"],
                    "aspect_polarities": {},
                    "return_risk_signal": False,
                }
            ],
            product_name="How comfortable is it to run with the Saucony Endorphin running shoes?",
        )

        aspects = {row["aspect"] for row in payload["counterfactuals"]}
        self.assertIn("fit", aspects)
        self.assertIn("comfort", aspects)
        self.assertNotIn("style", aspects)
        self.assertTrue(all("If the product had been redesigned" in row["counterfactual"] for row in payload["counterfactuals"]))
        self.assertTrue(all("heel slips" in row["observed_evidence"] or "harsh" in row["observed_evidence"] for row in payload["counterfactuals"]))
        self.assertGreaterEqual(payload["summary"]["counterfactual_count"], 2)

    def test_counterfactuals_ignore_positive_or_filter_contexts(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "filters",
                    "title": "RunRepeat filters",
                    "text": (
                        "Arch support Neutral. Midsole softness Soft Balanced Firm. Width / Fit Narrow Medium. "
                        "Condition Plantar fasciitis. Knee pain. Outsole durability Good."
                    ),
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": 0.0,
                    "aspects": [],
                    "aspect_polarities": {},
                },
                {
                    "feedback_id": "positive",
                    "title": "Positive ride",
                    "text": (
                        "Long runs feel forgiving, and I walk in the door afterwards with legs that don't feel beaten up. "
                        "That low leg fatigue after hard sessions is one of the most incredible things about this shoe. "
                        "The racing fit upper locks in and the platform feels trustworthy."
                    ),
                    "rating": 5,
                    "sentiment": "positive",
                    "sentiment_score": 0.8,
                    "aspects": ["comfort", "fit"],
                    "aspect_polarities": {"comfort": "positive", "fit": "positive"},
                },
            ],
            product_name="Saucony Endorphin",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)

    def test_counterfactuals_ignore_positive_review_sections(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "elite3",
                    "title": "Smoother, more planted",
                    "text": (
                        "I'd still classify it firmly as a neutral shoe, and runners with significant pronation "
                        "should look elsewhere, but the platform feels less tippy than the outgoing model. "
                        "Pair that with the wide footprint, and this version feels more trustworthy. "
                        "Racing fit upper that locks in and breathes well."
                    ),
                    "rating": 5,
                    "sentiment": "positive",
                    "sentiment_score": 0.7,
                    "aspects": ["fit", "comfort"],
                    "aspect_polarities": {"fit": "positive", "comfort": "positive"},
                },
                {
                    "feedback_id": "pro5",
                    "title": "What we like",
                    "text": (
                        "What we like about the Saucony Endorphin Pro 5: the step-in feel is comfortable. "
                        "The overall fit works well, it's secure but still accommodating and true to size. "
                        "The flat, stretchy tongue and laces provide solid lockdown."
                    ),
                    "rating": 5,
                    "sentiment": "positive",
                    "sentiment_score": 0.9,
                    "aspects": ["fit", "comfort"],
                    "aspect_polarities": {"fit": "positive", "comfort": "positive"},
                },
            ],
            product_name="Saucony Endorphin",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)

    def test_counterfactuals_validate_upstream_negative_aspects(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "upstream-noise",
                    "title": "Noisy extracted article",
                    "text": (
                        "Width / Fit Narrow (5) Medium (12). Midsole softness Soft Balanced Firm. "
                        "What we like: the step-in feel is comfortable, the fit works well, and the upper locks in."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.4,
                    "aspects": ["fit", "comfort"],
                    "aspect_polarities": {"fit": "negative", "comfort": "negative"},
                }
            ],
            product_name="Saucony Endorphin",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)

    def test_counterfactuals_ignore_positive_sofa_comfort_sections(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "movie-night",
                    "title": "Honest Lovesac review",
                    "text": (
                        "We ordered 4 recliners so we can each have one. "
                        "Movie night has never been more comfortable. The color and fabric work."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.3,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "negative"},
                },
                {
                    "feedback_id": "paint-palettes",
                    "title": "Five year couch review",
                    "text": (
                        "Here are some other topics you might be interested in: "
                        "Professionally curated color palettes and paint colors from Benjamin Moore."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.3,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "negative"},
                },
                {
                    "feedback_id": "plush",
                    "title": "Architectural Digest couch review",
                    "text": (
                        "The fabric doesn't look like the kind our dogs' nails will accidentally catch on. "
                        "It's all very plush, so I imagine it would be hard for kids to get hurt on it."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.3,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "negative"},
                },
            ],
            product_name="Lovesac sectional sofa",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)

    def test_counterfactuals_keep_real_sofa_comfort_complaints(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "bad-cushions",
                    "title": "Too firm",
                    "text": "The main complaint is that the cushions are too firm and uncomfortable after a long movie.",
                    "rating": 2,
                    "sentiment": "negative",
                    "sentiment_score": -0.6,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "negative"},
                }
            ],
            product_name="Lovesac sectional sofa",
        )

        self.assertGreaterEqual(payload["summary"]["counterfactual_count"], 1)
        self.assertEqual(payload["counterfactuals"][0]["aspect"], "comfort")

    def test_counterfactuals_promote_tense_gb_overlap_for_query_aspect(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "positive-comfort",
                    "title": "Honest Lovesac review",
                    "text": "The sectional is comfortable overall, and the modular layout worked well in our room.",
                    "rating": None,
                    "sentiment": "positive",
                    "sentiment_score": 0.5,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "positive"},
                }
            ],
            prometheus_twm={
                "gluing_diagnostics": [
                    {
                        "source_context": "chart:comfort",
                        "target_context": "chart:return_satisfaction",
                        "overlap_sections": 2,
                        "overlap_confidence": 0.91,
                        "weighted_glue_loss": 0.12,
                        "glue_loss": 0.13,
                        "compatible": False,
                    }
                ]
            },
            product_name="How comfortable is the Lovesac sectional sofa?",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 1)
        self.assertEqual(payload["summary"]["exploratory_count"], 0)
        self.assertEqual(payload["counterfactuals"][0]["aspect"], "comfort")
        self.assertEqual(payload["counterfactuals"][0]["evidence_status"], "gluing_supported")
        self.assertEqual(payload["counterfactuals"][0]["support_tier"], "gb-tense")

    def test_counterfactuals_do_not_promote_single_section_non_satisfaction_gb_overlap(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "positive-taste",
                    "title": "Amedei review",
                    "text": "The chocolate tastes balanced, aromatic, and refined.",
                    "rating": None,
                    "sentiment": "positive",
                    "sentiment_score": 0.5,
                    "aspects": ["taste"],
                    "aspect_polarities": {"taste": "positive"},
                }
            ],
            prometheus_twm={
                "gluing_diagnostics": [
                    {
                        "source_context": "corpus",
                        "target_context": "food",
                        "overlap_sections": 1,
                        "overlap_confidence": 0.91,
                        "weighted_glue_loss": 0.09,
                        "compatible": False,
                    }
                ]
            },
            product_name="How tasty is it to eat the Amedei Porcelana Chocolate Bars?",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)
        self.assertEqual(payload["summary"]["exploratory_count"], 1)
        self.assertEqual(payload["exploratory_counterfactuals"][0]["evidence_status"], "query_conditioned")

    def test_counterfactuals_map_sofa_aspects_to_topos_context_aliases(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        topos_psr = {
            "local_hankel_family": [
                {"context_id": "assemble"},
                {"context_id": "sit"},
            ],
            "restriction_diagnostics": [
                {"source_context": "post_purchase", "target_context": "assemble", "compatible": True},
                {"source_context": "use", "target_context": "sit", "compatible": False},
            ],
        }
        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "assembly",
                    "title": "Heavy setup",
                    "text": "For someone who cannot lift heavy objects, assembly could definitely be an issue.",
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["assembly"],
                    "aspect_polarities": {"assembly": "negative"},
                },
                {
                    "feedback_id": "comfort",
                    "title": "Comfort worry",
                    "text": "The main issue is that the sofa would eventually become flat and uncomfortable.",
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "negative"},
                },
            ],
            topos_psr=topos_psr,
            product_name="Lovesac sectional sofa",
        )

        rows_by_aspect = {
            row["aspect"]: row
            for row in [*payload["counterfactuals"], *payload["exploratory_counterfactuals"]]
        }
        self.assertEqual(rows_by_aspect["assembly"]["topos_status"], "overlap_stable")
        self.assertEqual(rows_by_aspect["comfort"]["topos_status"], "context_tense")

    def test_counterfactuals_do_not_explore_sofa_style_from_replacement_context(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "old-couch",
                    "title": "An Honest LoveSac Sectional Review",
                    "text": (
                        "For an eight-year-old, she doesn't look too shabby. Also, if you are researching "
                        "couches and you like this style, this sectional is from Crate and Barrel. "
                        "Old Couch Issues. Needless to say, it was time to start the replacement process."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.1,
                    "aspects": ["style"],
                    "aspect_polarities": {"style": "negative"},
                }
            ],
            product_name="Lovesac sectional sofa",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)
        self.assertEqual(payload["summary"]["exploratory_count"], 0)

    def test_counterfactuals_explore_sofa_value_from_price_context(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "price",
                    "title": "An Honest LoveSac Sectional Review - Is It Worth The Hefty PriceTag?",
                    "text": "The couch has many options, but the price felt hefty for the comfort we got.",
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["value"],
                    "aspect_polarities": {"value": "negative"},
                }
            ],
            product_name="Lovesac sectional sofa",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)
        self.assertEqual(payload["summary"]["exploratory_count"], 1)
        self.assertEqual(payload["exploratory_counterfactuals"][0]["aspect"], "value")

    def test_counterfactuals_do_not_explore_sofa_value_from_title_grade_or_header_context(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "title-only",
                    "title": "An Honest LoveSac Sectional Review - Is It Worth The Hefty PriceTag?",
                    "text": "The couch has many options and the research process took a while.",
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["value"],
                    "aspect_polarities": {"value": "negative"},
                },
                {
                    "feedback_id": "duplicated-title-intro",
                    "title": "An Honest LoveSac Sectional Review - Is It Worth The Hefty PriceTag?",
                    "text": (
                        "An Honest LoveSac Sectional Review - Is It Worth The Hefty PriceTag? "
                        "Have you been eyeing a new couch and want a REAL DEAL Lovesac sectional review? "
                        "Well, I just bought and installed one and I have a lot to say about it. "
                        "If you're on the fence, this just might help you decide."
                    ),
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["value"],
                    "aspect_polarities": {"value": "negative"},
                },
                {
                    "feedback_id": "ad-grade",
                    "title": "Couch Review: Lovesac Sactional - Architectural Digest",
                    "text": (
                        "The only change I would make would be to skip the surround sound system. "
                        "Sectional grades Comfort: A Value: B Delivery and assembly: F Style: B. "
                        "Lovesac Sactional, 5 Seats + 5 Sides $6,575 $3,945 (40% off)."
                    ),
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["value"],
                    "aspect_polarities": {"value": "negative"},
                },
                {
                    "feedback_id": "sheknows-header",
                    "title": "Honest Review of Lovesac After 2 Years - SheKnows",
                    "text": (
                        "Skip to main content Skip to header navigation View All February 18, 2026 "
                        "Share on Facebook Share on X Google Preferred."
                    ),
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["value"],
                    "aspect_polarities": {"value": "negative"},
                },
            ],
            product_name="Lovesac sectional sofa",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)
        self.assertEqual(payload["summary"]["exploratory_count"], 0)

    def test_counterfactuals_add_query_conditioned_fallback_for_comfort_queries(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "intro",
                    "title": "An Honest LoveSac Sectional Review",
                    "text": (
                        "Have you been eyeing a new couch and want a real deal Lovesac sectional review? "
                        "I bought and installed one and have a lot to say about it."
                    ),
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": 0.0,
                    "aspects": [],
                    "aspect_polarities": {},
                }
            ],
            product_name="How comfortable is the Lovesac sectional sofa?",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)
        self.assertEqual(payload["summary"]["exploratory_count"], 1)
        self.assertEqual(payload["exploratory_counterfactuals"][0]["aspect"], "comfort")
        self.assertEqual(payload["exploratory_counterfactuals"][0]["evidence_status"], "query_conditioned")

    def test_counterfactuals_suppress_shoe_fit_repairs_for_food_queries(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "chocablog",
                    "title": "Amedei Porcelana",
                    "text": (
                        "The individual pieces are perhaps slightly too big for my liking though. "
                        "The aroma is rich and chocolatey, but there's nothing that stands out about it."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.2,
                    "aspects": ["fit"],
                    "aspect_polarities": {"fit": "negative"},
                }
            ],
            product_name="How tasty is it to eat the Amedei Porcelana Chocolate Bars?",
        )

        aspects = {row["aspect"] for row in payload["counterfactuals"]}
        exploratory_aspects = {row["aspect"] for row in payload["exploratory_counterfactuals"]}
        self.assertNotIn("fit", aspects)
        self.assertIn("taste", exploratory_aspects)
        self.assertEqual(payload["exploratory_counterfactuals"][0]["evidence_status"], "domain_plausible")

    def test_counterfactuals_do_not_treat_food_marketing_flavor_as_negative_taste(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "amedei-shop",
                    "title": "Porcelain: dark chocolate 70% Limited - Amedei shop",
                    "text": (
                        "This precious cocoa creates a light, harmonious, and aromatic chocolate whose "
                        "uniqueness is evident from the first taste. Notes of almond, toasted bread and bay leaf."
                    ),
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": 0.1,
                    "aspects": ["taste"],
                    "aspect_polarities": {"taste": "negative"},
                }
            ],
            product_name="Amedei Porcelana Chocolate Bars",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)
        self.assertEqual(payload["summary"]["exploratory_count"], 0)

    def test_counterfactuals_use_query_conditioned_food_probe_when_no_negative_taste_state(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "amedei-shop",
                    "title": "Porcelain: dark chocolate 70% Limited - Amedei shop",
                    "text": (
                        "This precious cocoa creates a light, harmonious, and aromatic chocolate whose "
                        "uniqueness is evident from the first taste. Notes of almond, toasted bread and bay leaf."
                    ),
                    "rating": None,
                    "sentiment": "mixed",
                    "sentiment_score": 0.1,
                    "aspects": ["taste"],
                    "aspect_polarities": {"taste": "negative"},
                }
            ],
            product_name="How tasty is it to eat the Amedei Porcelana Chocolate Bars?",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)
        self.assertEqual(payload["summary"]["exploratory_count"], 1)
        self.assertEqual(payload["exploratory_counterfactuals"][0]["aspect"], "taste")
        self.assertEqual(payload["exploratory_counterfactuals"][0]["evidence_status"], "query_conditioned")

    def test_counterfactuals_keep_real_food_taste_complaints(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "taste",
                    "title": "Flat chocolate",
                    "text": "The main complaint is that the chocolate tastes waxy and too bitter.",
                    "rating": 2,
                    "sentiment": "negative",
                    "sentiment_score": -0.7,
                    "aspects": ["taste"],
                    "aspect_polarities": {"taste": "negative"},
                }
            ],
            product_name="Amedei Porcelana Chocolate Bars",
        )

        self.assertGreaterEqual(payload["summary"]["counterfactual_count"], 1)
        self.assertEqual(payload["counterfactuals"][0]["aspect"], "taste")

    def test_counterfactuals_keep_food_taste_disappointment_as_supported(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "chocablog-disappointment",
                    "title": "Amedei Porcelana - Chocablog",
                    "text": (
                        "I hoped for the same wonderful porcelana taste as I was used to from another brand, "
                        "but unfortunately discovered that Amedei is my big disappointment. I was even shocked."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.4,
                    "aspects": ["taste"],
                    "aspect_polarities": {"taste": "negative"},
                }
            ],
            product_name="How tasty is it to eat the Amedei Porcelana Chocolate Bars?",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 1)
        self.assertEqual(payload["summary"]["exploratory_count"], 0)
        self.assertEqual(payload["counterfactuals"][0]["aspect"], "taste")
        self.assertEqual(payload["counterfactuals"][0]["evidence_status"], "supported")
        self.assertEqual(payload["counterfactuals"][0]["support_tier"], "moderate")

    def test_dashboard_regenerates_stale_counterfactual_artifact(self) -> None:
        try:
            from functorflow_v3.product_feedback_visualizations import generate_product_feedback_dashboard
        except ImportError:
            from ..functorflow_v3.product_feedback_visualizations import generate_product_feedback_dashboard

        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            (run_path / "product_success_scorecard.json").write_text(
                json.dumps(
                    {
                        "product_name": "Lovesac sectional sofa",
                        "brand_name": "",
                        "verdict": "mixed_positive",
                        "top_negative_aspects": [],
                        "top_positive_aspects": [],
                        "top_return_risk_aspects": [],
                    }
                ),
                encoding="utf-8",
            )
            for filename, payload in {
                "outcome_summary.json": {"feedback_count": 1},
                "aspect_summary.json": {"aspect_summary": {}},
                "causal_hypotheses.json": {"hypotheses": []},
                "usage_workflows.json": {"top_workflow_motifs": []},
                "ablation_comparison.json": {"rows": [], "takeaways": []},
                "topos_psr_hankel.json": {"summary": {}, "local_hankel_family": [], "restriction_diagnostics": []},
            }.items():
                (run_path / filename).write_text(json.dumps(payload), encoding="utf-8")
            (run_path / "normalized_feedback.jsonl").write_text(
                json.dumps(
                    {
                        "feedback_id": "positive-comfort",
                        "title": "Positive comfort",
                        "text": "Movie night has never been more comfortable.",
                        "sentiment": "negative",
                        "sentiment_score": -0.2,
                        "aspects": ["comfort"],
                        "aspect_polarities": {"comfort": "negative"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_path / "prometheus_counterfactuals.json").write_text(
                json.dumps(
                    {
                        "schema_version": "cliff.prometheus_counterfactuals.v1",
                        "summary": {"counterfactual_count": 1},
                        "counterfactuals": [
                            {
                                "title": "stale",
                                "aspect": "comfort",
                                "counterfactual": "stale repair",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            generate_product_feedback_dashboard(run_path)

            regenerated = json.loads((run_path / "prometheus_counterfactuals.json").read_text(encoding="utf-8"))
            self.assertEqual(regenerated["summary"]["counterfactual_count"], 0)

    def test_prometheus_gluing_rows_prioritize_learned_edges(self) -> None:
        try:
            from functorflow_v3 import product_feedback_visualizations as viz
        except ImportError:
            from ..functorflow_v3 import product_feedback_visualizations as viz

        html = viz._prometheus_gluing_rows(
            [
                {
                    "source_context": "corpus",
                    "target_context": "chart:comfort",
                    "construction": "corpus_projection",
                    "overlap_sections": 1,
                    "overlap_confidence": 0.96,
                    "weighted_glue_loss": 0.04,
                    "compatible": True,
                },
                {
                    "source_context": "chart:activity_use",
                    "target_context": "chart:return_satisfaction",
                    "construction": "learned_overlap",
                    "overlap_sections": 2,
                    "overlap_confidence": 0.82,
                    "weighted_glue_loss": 0.12,
                    "compatible": False,
                },
            ]
        )

        self.assertLess(html.index("chart:activity_use"), html.index("corpus"))
        self.assertIn("learned overlap", html)

    def test_counterfactuals_keep_explicit_complaints(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "complaint",
                    "title": "Heel issue",
                    "text": "The main drawback is that the heel slips on faster runs and the upper rubs my foot.",
                    "rating": 3,
                    "sentiment": "negative",
                    "sentiment_score": -0.4,
                    "aspects": [],
                    "aspect_polarities": {},
                }
            ],
            product_name="Saucony Endorphin",
        )

        aspects = {row["aspect"] for row in payload["counterfactuals"]}
        self.assertIn("fit", aspects)

    def test_counterfactuals_ignore_running_shoe_navigation_and_filter_snippets(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "filters",
                    "title": "RunRepeat collection",
                    "text": (
                        "Arch support Neutral (17) Breathability Breathable (8) Midsole softness Soft (2) "
                        "Balanced (4) Firm (1) Condition Plantar fasciitis (1) Knee pain (1) "
                        "Width / Fit Narrow (5) Medium (12) Toebox width Narrow (3) Medium (6)."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.4,
                    "aspects": ["fit", "comfort"],
                    "aspect_polarities": {"fit": "negative", "comfort": "negative"},
                },
                {
                    "feedback_id": "nav",
                    "title": "Doctors of Running",
                    "text": (
                        "ASICS Metaspeed Edge Tokyo Review (2025) Plantar Fasciitis, Revisited | DOR Podcast. "
                        "Mailbag! Plated Trainers and Achilles Pain? We Rank Our Top 5 Comfortable Running Shoes."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.4,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "negative"},
                },
                {
                    "feedback_id": "menu",
                    "title": "Believe in the Run menu",
                    "text": (
                        "Shop BITR Apparel Shop Now Shoes Gear News Events Videos Podcasts Nutrition & Training "
                        "Shoe Finder Road Best Of Daily Trainers Tempo Race Day Max Cushion Wide Foot Lifestyle "
                        "Trail Best Of Technical Non-Technical Long Distance Race Day Wide Foot Track/XC Best Of "
                        "Spikes Flats Most Popular Brands New Balance Nike Asics Hoka Saucony Puma Adidas Anatomy of."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.4,
                    "aspects": ["fit"],
                    "aspect_polarities": {"fit": "negative"},
                },
                {
                    "feedback_id": "comparative",
                    "title": "Comparative aside",
                    "text": (
                        "Sometimes normal is all you want from a running shoe, far better than an expensive shoe "
                        "that tries to do too much and ends up falling flat because they're too uncomfortable to wear."
                    ),
                    "rating": None,
                    "sentiment": "negative",
                    "sentiment_score": -0.4,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "negative"},
                },
            ],
            product_name="Saucony Endorphin running shoes",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 0)

    def test_counterfactuals_keep_heel_rubbing_cons(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "heel",
                    "title": "Best Carbon Shoes",
                    "text": "Pros lightweight and speedy. Cons irritation on the back of the heel caused by rubbing.",
                    "rating": 4,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["fit"],
                    "aspect_polarities": {"fit": "negative"},
                }
            ],
            product_name="Saucony Endorphin running shoes",
        )

        self.assertGreaterEqual(payload["summary"]["counterfactual_count"], 1)
        self.assertEqual(payload["counterfactuals"][0]["aspect"], "fit")
        self.assertEqual(payload["counterfactuals"][0]["support_tier"], "strong")

    def test_counterfactuals_keep_mixed_positive_stability_limitations_as_exploratory(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "stability",
                    "title": "GearLab",
                    "text": (
                        "While it may not be ideal for correcting unstable gait patterns, it provides enough "
                        "stability for long runs, promotes agility, and offers a reliable, strong feel."
                    ),
                    "rating": 4,
                    "sentiment": "mixed",
                    "sentiment_score": -0.1,
                    "aspects": ["traction"],
                    "aspect_polarities": {"traction": "negative"},
                }
            ],
            product_name="Saucony Endorphin running shoes",
        )

        self.assertEqual(payload["summary"]["counterfactual_count"], 1)
        self.assertEqual(payload["counterfactuals"][0]["support_tier"], "exploratory")

    def test_counterfactuals_keep_technical_plate_problem_as_moderate(self) -> None:
        try:
            from functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals
        except ImportError:
            from ..functorflow_v3.product_feedback_counterfactuals import build_product_feedback_counterfactuals

        payload = build_product_feedback_counterfactuals(
            [
                {
                    "feedback_id": "plate",
                    "title": "Plate stiffness",
                    "text": (
                        "If someone finds the plate too stiff in the sagittal plane, it will push them even "
                        "farther in the frontal plane. This will create a major problem for efficiency."
                    ),
                    "rating": 3,
                    "sentiment": "negative",
                    "sentiment_score": -0.2,
                    "aspects": ["comfort"],
                    "aspect_polarities": {"comfort": "negative"},
                }
            ],
            product_name="Saucony Endorphin running shoes",
        )

        self.assertGreaterEqual(payload["summary"]["counterfactual_count"], 1)
        self.assertEqual(payload["counterfactuals"][0]["support_tier"], "moderate")

    def test_article_rating_extraction_handles_percent_and_fraction_formats(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        raw, scale = module._extract_article_rating_from_text(
            "Nike Pegasus 41 Review",
            "OUR VERDICT: 79% - GOOD",
        )
        self.assertEqual((raw, scale), (79.0, 100.0))

        raw, scale = module._extract_article_rating_from_text(
            "Lovesac Sactionals Sofa Review",
            "Product Overview Sofa Overall Score 4.2/5 Pros Cons",
        )
        self.assertEqual((raw, scale), (4.2, 5.0))

        raw, scale = module._extract_article_rating_from_text(
            "Lovesac Sactionals Sofa Review",
            "Product Overview Sofa Overall Score Pros Cons Ideal For Lovesac Sactionals Sofa 4.2/5 Modular layout options",
        )
        self.assertEqual((raw, scale), (4.2, 5.0))

    def test_article_rating_extraction_handles_expert_score_phrase(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        raw, scale = module._extract_article_rating_from_text(
            "Nike Pegasus 41 review",
            "Nike Pegasus 41 review 7 expert score 7 user's score",
        )
        self.assertEqual((raw, scale), (7.0, 10.0))

    def test_article_rating_extraction_handles_out_of_stars_phrase(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        raw, scale = module._extract_article_rating_from_text(
            "My Lovesac Sactionals Review",
            "I give it 5 out of 5 stars for the form, function and comfort and I'd order it all over again!",
        )
        self.assertEqual((raw, scale), (5.0, 5.0))

    def test_workflow_parallel_frontiers_reflect_feedback_pipeline(self) -> None:
        workflow = build_product_feedback_agentic_workflow()

        frontiers = tuple(tuple(agent.name for agent in frontier) for frontier in workflow.parallel_frontiers())

        self.assertEqual(frontiers[0], ("feedback_collection_agent",))
        self.assertEqual(frontiers[1], ("feedback_normalization_agent",))
        self.assertEqual(frontiers[2], ("usage_workflow_agent", "aspect_grounding_agent", "outcome_signal_agent"))
        self.assertEqual(frontiers[3], ("causal_hypothesis_agent",))
        self.assertEqual(frontiers[4], ("success_scoring_agent",))
        self.assertEqual(frontiers[5], ("ablation_comparison_agent",))
        self.assertEqual(frontiers[-1], ("executive_summary_agent",))

    def test_runner_flags_fit_risk_and_recommend_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "feedback.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "r1",
                                "title": "Love the comfort",
                                "text": "Very comfortable and stylish. True to size and easy to slip on.",
                                "rating": 5,
                                "source": "reviews",
                                "source_reference": "https://example.com/review/r1",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r2",
                                "title": "Too tight",
                                "text": "These were too tight in the toe box and I returned them.",
                                "rating": 2,
                                "source": "reviews",
                                "returned": True,
                                "source_reference": "https://example.com/review/r2",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r3",
                                "title": "Runs narrow",
                                "text": "Nice style but the fit runs small and narrow. I had to send it back.",
                                "rating": 1,
                                "source": "qna",
                                "source_reference": "https://example.com/review/r3",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r4",
                                "title": "Convenient for travel",
                                "text": "Easy to slip on and comfortable for airport use.",
                                "rating": 4,
                                "source": "social",
                                "source_reference": "https://example.com/review/r4",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r5",
                                "title": "Not worth the price",
                                "text": "Overpriced and poor quality for the money.",
                                "rating": 2,
                                "source": "reviews",
                                "source_reference": "https://example.com/review/r5",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Slip-On Sneaker",
                    brand_name="Amazon Basics",
                )
            )
            result = runner.run()

            self.assertTrue(result.success_scorecard_path.exists())
            self.assertTrue(result.usage_workflows_path.exists())
            self.assertTrue(result.ablation_comparison_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.dashboard_path.exists())
            self.assertTrue(result.dashboard_summary_path.exists())
            self.assertIsNotNone(result.review_episodes_path)
            self.assertIsNotNone(result.topos_psr_path)
            self.assertIsNotNone(result.prometheus_counterfactuals_path)
            self.assertIsNotNone(result.prometheus_twm_path)
            self.assertIsNotNone(result.prometheus_twm_report_path)
            self.assertTrue(result.review_episodes_path.exists())
            self.assertTrue(result.topos_psr_path.exists())
            self.assertTrue(result.prometheus_counterfactuals_path.exists())
            self.assertTrue(result.prometheus_twm_path.exists())
            self.assertTrue(result.prometheus_twm_report_path.exists())

            scorecard = json.loads(result.success_scorecard_path.read_text(encoding="utf-8"))
            self.assertEqual(scorecard["verdict"], "at_risk")
            self.assertTrue(scorecard["return_warning_recommended"])
            self.assertIn("fit", scorecard["top_return_risk_aspects"])
            self.assertIn("comfort", scorecard["top_positive_aspects"])

            hypotheses = json.loads(result.causal_hypotheses_path.read_text(encoding="utf-8"))
            hypothesis_sources = {item["src"] for item in hypotheses["hypotheses"]}
            self.assertIn("tight or inconsistent fit perception", hypothesis_sources)
            self.assertIn("run-time usage friction", hypothesis_sources)

            usage_workflows = json.loads(result.usage_workflows_path.read_text(encoding="utf-8"))
            top_motifs = [" -> ".join(row["workflow_stages"]) for row in usage_workflows["top_workflow_motifs"]]
            self.assertTrue(any("wear" in motif or "run" in motif for motif in top_motifs))

            ablation = json.loads(result.ablation_comparison_path.read_text(encoding="utf-8"))
            ablation_labels = [row["label"] for row in ablation["rows"]]
            self.assertEqual(ablation_labels[:2], ["Prompt-like baseline", "BAFFLE structured scaffold"])
            self.assertTrue(any("score delta" in item.lower() for item in ablation["takeaways"]))

            report = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Often returned due to fit issues", report)
            self.assertIn("Which product aspects most strongly appear to drive return risk?", report)
            self.assertIn("Ablation Comparison", report)
            self.assertIn("Quantitative Takeaways", report)
            self.assertIn("Usage Workflows", report)
            self.assertIn("Topos PSR", report)
            self.assertIn("Prometheus Counterfactuals", report)

            counterfactuals = json.loads(result.prometheus_counterfactuals_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(counterfactuals["summary"]["counterfactual_count"], 1)
            self.assertTrue(any(row["aspect"] == "fit" for row in counterfactuals["counterfactuals"]))

            topos_psr = json.loads(result.topos_psr_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(topos_psr["summary"]["n_contexts"], 1)
            prometheus_twm = json.loads(result.prometheus_twm_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(prometheus_twm["summary"]["local_psr_count"], 1)
            self.assertIn("twm_objective_glue_term", prometheus_twm["summary"])

            dashboard = result.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("CLIFF Product Feedback Dashboard", dashboard)
            self.assertIn("Outcome Snapshot", dashboard)
            self.assertIn("Ablation Comparison", dashboard)
            self.assertIn("Prompt-like baseline", dashboard)
            self.assertIn("Causal Hypotheses", dashboard)
            self.assertIn("Usage Workflows", dashboard)
            self.assertIn("Topos PSR", dashboard)
            self.assertIn("Prometheus Counterfactuals", dashboard)
            self.assertIn("Prometheus Topos World Model", dashboard)
            self.assertIn("Learned Overlap Edges", dashboard)
            self.assertIn("GB Glue Term", dashboard)
            self.assertIn("Supported Repairs", dashboard)
            self.assertIn("Low-Confidence Repair Probes", dashboard)
            self.assertIn("Counterfactual Intervention", dashboard)
            self.assertIn("Evidence Preview", dashboard)
            self.assertIn("Open source", dashboard)

            dashboard_summary = json.loads(result.dashboard_summary_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(int(dashboard_summary["topos_context_count"]), 1)
            self.assertGreaterEqual(int(dashboard_summary["prometheus_counterfactual_count"]), 1)
            self.assertEqual(dashboard_summary["prometheus_twm_status"], "ok")
            self.assertGreaterEqual(int(dashboard_summary["prometheus_twm_local_psr_count"]), 1)

    def test_runner_preserves_source_reference_in_normalized_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "feedback.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "r1",
                        "title": "Comfort review",
                        "text": "Comfortable and easy to use.",
                        "rating": 4,
                        "source": "reviews",
                        "source_reference": "https://example.com/review-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Demo Product",
                    brand_name="Demo Brand",
                )
            )
            result = runner.run()

            rows = [
                json.loads(line)
                for line in result.normalized_feedback_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(rows[0]["source_reference"], "https://example.com/review-1")

    def test_runner_builds_vehicle_workflows_without_sofa_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "vehicle_feedback.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "v1",
                                "title": "Great commuter EV",
                                "text": "Easy to drive in traffic, comfortable on long commutes, and charging at home is simple.",
                                "rating": 5,
                                "source": "reviews",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "v2",
                                "title": "Road trip favorite",
                                "text": "The Tesla Model 3 handles well, the steering feels precise, and Supercharger stops are convenient.",
                                "rating": 4,
                                "source": "reviews",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Tesla Model 3",
                    brand_name="Tesla",
                )
            )
            result = runner.run()

            usage_workflows = json.loads(result.usage_workflows_path.read_text(encoding="utf-8"))
            top_motifs = [" -> ".join(row["workflow_stages"]) for row in usage_workflows["top_workflow_motifs"]]
            self.assertTrue(any("drive" in motif for motif in top_motifs))
            self.assertTrue(any("charge" in motif for motif in top_motifs))
            self.assertTrue(all("assemble" not in motif for motif in top_motifs))
            self.assertTrue(all("sit" not in motif for motif in top_motifs))
            self.assertTrue(all("wash" not in motif for motif in top_motifs))

    def test_runner_builds_food_workflows_without_sofa_or_vehicle_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "food_feedback.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "f1",
                                "title": "Elegant chocolate",
                                "text": "We opened the bar after dinner, tasted rich cocoa notes, and happily shared the rest.",
                                "rating": 5,
                                "source": "reviews",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "f2",
                                "title": "Good but worth storing carefully",
                                "text": "Delicious flavor and smooth texture. I ate a few squares, then stored the rest for later.",
                                "rating": 4,
                                "source": "reviews",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Amedei Porcelana Chocolate Bars",
                    brand_name="Amedei",
                )
            )
            result = runner.run()

            usage_workflows = json.loads(result.usage_workflows_path.read_text(encoding="utf-8"))
            self.assertEqual(usage_workflows["usage_family"], "food")
            top_motifs = [" -> ".join(row["workflow_stages"]) for row in usage_workflows["top_workflow_motifs"]]
            self.assertTrue(any("taste" in motif or "eat" in motif for motif in top_motifs))
            self.assertTrue(any("open" in motif or "share" in motif or "store" in motif for motif in top_motifs))
            self.assertTrue(all("sit" not in motif for motif in top_motifs))
            self.assertTrue(all("run" not in motif for motif in top_motifs))
            self.assertTrue(all("drive" not in motif for motif in top_motifs))
            self.assertTrue(all("assemble" not in motif for motif in top_motifs))

            dashboard = result.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("unwrap it, taste it, eat it, share it, store it", dashboard)

    def test_runner_normalizes_ratings_from_multiple_scales(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "feedback.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "r1",
                                "title": "Five-star equivalent",
                                "text": "Comfortable and durable.",
                                "rating": 4,
                                "rating_scale": 5,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r2",
                                "title": "Ten-point equivalent",
                                "text": "Comfortable and supportive.",
                                "rating": 8,
                                "rating_scale": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r3",
                                "title": "Percent equivalent",
                                "text": "Stylish and worth it.",
                                "rating": 80,
                                "rating_scale": 100,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Scaled Sneaker",
                    brand_name="FF2",
                )
            )
            result = runner.run()

            outcome = json.loads(result.outcome_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(outcome["average_rating"], 4.0)

            normalized_rows = [
                json.loads(line)
                for line in result.normalized_feedback_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["rating"] for row in normalized_rows], [4.0, 4.0, 4.0])


if __name__ == "__main__":
    unittest.main()
