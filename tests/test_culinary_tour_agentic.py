"""Tests for the CLIFF culinary tour orchestrator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error as urllib_error

try:
    from functorflow_v3.culinary_tour_agentic import (
        CulinaryTourAgenticRunner,
        CulinaryTourLookupError,
        DiscoveredCulinaryStop,
        interpret_culinary_query,
    )
except ModuleNotFoundError:
    from ..functorflow_v3.culinary_tour_agentic import (
        CulinaryTourAgenticRunner,
        CulinaryTourLookupError,
        DiscoveredCulinaryStop,
        interpret_culinary_query,
    )


class CulinaryTourAgenticTests(unittest.TestCase):
    def test_runner_uses_explicit_manifest_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "stops.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "name": "Seoul Kimchi Lab",
                                "destination": "Seoul",
                                "district": "Yongsan",
                                "specialty": "kimchi tasting menu and fermentation workshop",
                                "estimated_cost": 33,
                                "tags": ["kimchi", "fermentation"],
                                "url": "https://example.com/seoul/kimchi-lab",
                            }
                        ),
                        json.dumps(
                            {
                                "name": "Jongno Kimchi Table",
                                "destination": "Seoul",
                                "district": "Jongno",
                                "specialty": "kimchi stew and banchan",
                                "estimated_cost": 24,
                                "tags": ["kimchi", "stew"],
                                "url": "https://example.com/seoul/jongno-kimchi-table",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = CulinaryTourAgenticRunner(
                "Plan a kimchi culinary tour in Seoul July 6-11 under $50 per meal",
                workdir / "culinary",
                manifest_path=manifest_path,
            ).run()

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["retrieval_backend"], "manifest")
            self.assertEqual(summary["culinary_manifest_path"], str(manifest_path.resolve()))

    def test_interpret_culinary_query_extracts_core_constraints(self) -> None:
        plan = interpret_culinary_query("Plan a kimchi culinary tour in Seoul July 6-11 under $50 per meal")

        self.assertEqual(plan.food_focus.lower(), "kimchi")
        self.assertEqual(plan.destination, "Seoul")
        self.assertEqual(plan.budget_per_meal, 50)
        self.assertEqual(plan.estimated_days, 6)

    def test_interpret_culinary_query_handles_from_to_date_language(self) -> None:
        plan = interpret_culinary_query("Plan a kimchi tour of Seoul from July 6 to 10th for under $50 per meal.")

        self.assertEqual(plan.food_focus.lower(), "kimchi")
        self.assertEqual(plan.destination, "Seoul")
        self.assertEqual(plan.time_window.lower(), "from july 6 to 10th")
        self.assertEqual(plan.budget_per_meal, 50)
        self.assertEqual(plan.estimated_days, 5)

    def test_interpret_culinary_query_handles_generic_city_and_cuisine(self) -> None:
        plan = interpret_culinary_query(
            "Arrange a dumplings tour in San Francisco Chinatown for the period May 1-4 where each meal is under $50"
        )

        self.assertEqual(plan.food_focus.lower(), "dumplings")
        self.assertEqual(plan.destination, "San Francisco Chinatown")
        self.assertEqual(plan.time_window, "May 1-4")
        self.assertEqual(plan.budget_per_meal, 50)
        self.assertEqual(plan.estimated_days, 4)

    def test_interpret_culinary_query_handles_tour_for_destination_language(self) -> None:
        plan = interpret_culinary_query("Plan a seafood tour for Boston from July 5th-10th")

        self.assertEqual(plan.food_focus.lower(), "seafood")
        self.assertEqual(plan.destination, "Boston")
        self.assertEqual(plan.time_window, "from July 5th-10th")
        self.assertEqual(plan.estimated_days, 6)

    def test_runner_emits_dashboard_itinerary_and_broadcasts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            live_stops = [
                DiscoveredCulinaryStop(
                    stop_id="osm_node_1",
                    name="Kimchi House",
                    destination="Seoul",
                    district="Jongno",
                    specialty="kimchi and fermentation tasting",
                    estimated_cost=24,
                    tags=("kimchi", "fermentation"),
                    url="https://example.com/kimchi-house",
                    source_type="osm_live",
                ),
                DiscoveredCulinaryStop(
                    stop_id="osm_node_2",
                    name="Hongdae Hansik",
                    destination="Seoul",
                    district="Hongdae",
                    specialty="kimchi stew and shared banchan",
                    estimated_cost=28,
                    tags=("kimchi", "hansik"),
                    url="https://example.com/hongdae-hansik",
                    source_type="osm_live",
                ),
            ]
            with patch("functorflow_v3.culinary_tour_agentic._live_stop_candidates", return_value=live_stops):
                result = CulinaryTourAgenticRunner(
                    "Plan a kimchi culinary tour in Seoul July 6-11 under $50 per meal",
                    Path(tmpdir) / "culinary",
                ).run()

            broadcasts = json.loads(result.broadcasts_path.read_text(encoding="utf-8"))
            itinerary = json.loads(result.itinerary_path.read_text(encoding="utf-8"))
            dashboard_html = result.dashboard_path.read_text(encoding="utf-8")

            self.assertGreaterEqual(len(broadcasts["broadcasts"]), 6)
            self.assertGreaterEqual(len(itinerary["itinerary"]), 2)
            self.assertIn("kimchi", dashboard_html.lower())
            self.assertIn("conscious broadcasts", dashboard_html.lower())
            self.assertIn("later unconscious agents read earlier conscious broadcasts", dashboard_html.lower())
            self.assertIn("Read This in the Book", dashboard_html)
            self.assertIn("catagi.pdf", dashboard_html)
            self.assertEqual(itinerary["retrieval_backend"], "osm_live")
            self.assertTrue(any(item.get("source_agent") == "stop_retrieval_agent" for item in broadcasts["broadcasts"]))
            self.assertTrue(any(item.get("read_broadcast_ids") for item in broadcasts["broadcasts"][1:]))
            self.assertIn("read from broadcast-", dashboard_html.lower())
            self.assertIn("guided by broadcast-", dashboard_html.lower())
            self.assertIn("https://example.com/kimchi-house", dashboard_html)
            self.assertTrue(any(item.get("venue_url") for item in itinerary["itinerary"]))

    def test_runner_accepts_custom_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "stops.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "name": "Seoul Kimchi Lab",
                                "destination": "Seoul",
                                "district": "Yongsan",
                                "specialty": "kimchi tasting menu and fermentation workshop",
                                "estimated_cost": 33,
                                "tags": ["kimchi", "fermentation"],
                                "url": "https://example.com/seoul/kimchi-lab",
                            }
                        ),
                        json.dumps(
                            {
                                "name": "Seoul Market Ferment Studio",
                                "destination": "Seoul",
                                "district": "Jongno",
                                "specialty": "market kimchi, banchan, and fermentation flights",
                                "estimated_cost": 22,
                                "tags": ["kimchi", "market"],
                                "url": "https://example.com/seoul/market-ferment-studio",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            result = CulinaryTourAgenticRunner(
                "Plan a kimchi culinary tour in Seoul July 6-11 under $50 per meal",
                workdir / "culinary",
                manifest_path=manifest_path,
            ).run()

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["retrieval_backend"], "manifest")
            self.assertEqual(summary["culinary_manifest_path"], str(manifest_path.resolve()))

    def test_runner_uses_live_retrieval_for_generic_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            live_stops = [
                DiscoveredCulinaryStop(
                    stop_id="osm_node_100",
                    name="Good Mong Kok Bakery",
                    destination="San Francisco Chinatown",
                    district="Chinatown",
                    specialty="dumplings and dim sum",
                    estimated_cost=14,
                    tags=("dumplings", "dim sum"),
                    url="https://example.com/good-mong-kok",
                    source_type="osm_live",
                ),
                DiscoveredCulinaryStop(
                    stop_id="osm_node_101",
                    name="Yuanbao Jiaozi",
                    destination="San Francisco Chinatown",
                    district="Chinatown",
                    specialty="northern dumplings",
                    estimated_cost=22,
                    tags=("dumplings",),
                    url="https://example.com/yuanbao-jiaozi",
                    source_type="osm_live",
                ),
            ]
            with patch("functorflow_v3.culinary_tour_agentic._live_stop_candidates", return_value=live_stops):
                result = CulinaryTourAgenticRunner(
                    "Arrange a dumplings tour in San Francisco Chinatown for the period May 1-4 where each meal is under $50",
                    workdir / "culinary",
                ).run()

            itinerary = json.loads(result.itinerary_path.read_text(encoding="utf-8"))
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["retrieval_backend"], "osm_live")
            self.assertIsNone(summary["culinary_manifest_path"])
            self.assertEqual(itinerary["query_plan"]["food_focus"].lower(), "dumplings")
            self.assertTrue(
                all(
                    item["venue_name"] in {"Good Mong Kok Bakery", "Yuanbao Jiaozi"}
                    for item in itinerary["itinerary"]
                )
            )

    def test_runner_without_budget_does_not_invent_default_price_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            live_stops = [
                DiscoveredCulinaryStop(
                    stop_id="osm_node_300",
                    name="Kimchi House",
                    destination="Seoul",
                    district="Jongno",
                    specialty="kimchi and fermentation tasting",
                    estimated_cost=None,
                    tags=("kimchi",),
                    url="https://example.com/kimchi-house",
                    source_type="osm_live",
                ),
                DiscoveredCulinaryStop(
                    stop_id="osm_node_301",
                    name="Seoul Hansik Table",
                    destination="Seoul",
                    district="Mapo",
                    specialty="traditional Korean dishes",
                    estimated_cost=None,
                    tags=("korean",),
                    url="https://example.com/seoul-hansik-table",
                    source_type="osm_live",
                ),
            ]
            with patch("functorflow_v3.culinary_tour_agentic._live_stop_candidates", return_value=live_stops):
                result = CulinaryTourAgenticRunner(
                    "Arrange a kimchi tour of Seoul from July 6-10th",
                    workdir / "culinary",
                ).run()

            itinerary = json.loads(result.itinerary_path.read_text(encoding="utf-8"))
            self.assertIsNone(itinerary["query_plan"]["budget_per_meal"])
            self.assertEqual(itinerary["retrieval_backend"], "osm_live")

    def test_runner_fails_when_live_results_do_not_support_budget_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            live_stops = [
                DiscoveredCulinaryStop(
                    stop_id="osm_node_200",
                    name="Expensive Dumpling Club",
                    destination="San Francisco Chinatown",
                    district="Chinatown",
                    specialty="dumplings tasting menu",
                    estimated_cost=75,
                    tags=("dumplings",),
                    url="https://example.com/expensive-dumpling-club",
                    source_type="osm_live",
                )
            ]
            with patch("functorflow_v3.culinary_tour_agentic._live_stop_candidates", return_value=live_stops):
                with self.assertRaises(CulinaryTourLookupError) as ctx:
                    CulinaryTourAgenticRunner(
                        "Arrange a dumplings tour in San Francisco Chinatown for the period May 1-4 where each meal is under $50",
                        workdir / "culinary",
                    ).run()
            self.assertIn("explicit price data", str(ctx.exception))
            self.assertIn("within budget", str(ctx.exception))

    def test_load_overpass_json_retries_and_uses_fallback_endpoint(self) -> None:
        calls: list[str] = []

        def fake_loader(url, *, headers, data=None, timeout=20.0):
            del headers, data, timeout
            calls.append(url)
            if url == "https://primary.example/api/interpreter":
                raise urllib_error.HTTPError(url, 504, "Gateway Timeout", hdrs=None, fp=None)
            return {"elements": []}

        with patch("functorflow_v3.culinary_tour_agentic._overpass_endpoints", return_value=(
            "https://primary.example/api/interpreter",
            "https://fallback.example/api/interpreter",
        )):
            with patch("functorflow_v3.culinary_tour_agentic._load_json_from_url", side_effect=fake_loader):
                from functorflow_v3.culinary_tour_agentic import _load_overpass_json

                payload = _load_overpass_json("[out:json];node;out;")

        self.assertEqual(payload, {"elements": []})
        self.assertEqual(
            calls,
            [
                "https://primary.example/api/interpreter",
                "https://fallback.example/api/interpreter",
            ],
        )

    def test_load_overpass_json_retries_on_timeout_error(self) -> None:
        calls: list[str] = []

        def fake_loader(url, *, headers, data=None, timeout=20.0):
            del headers, data, timeout
            calls.append(url)
            if len(calls) == 1:
                raise TimeoutError("The read operation timed out")
            return {"elements": []}

        with patch("functorflow_v3.culinary_tour_agentic._overpass_endpoints", return_value=(
            "https://primary.example/api/interpreter",
            "https://fallback.example/api/interpreter",
        )):
            with patch("functorflow_v3.culinary_tour_agentic._load_json_from_url", side_effect=fake_loader):
                from functorflow_v3.culinary_tour_agentic import _load_overpass_json

                payload = _load_overpass_json("[out:json];node;out;")

        self.assertEqual(payload, {"elements": []})
        self.assertEqual(
            calls,
            [
                "https://primary.example/api/interpreter",
                "https://fallback.example/api/interpreter",
            ],
        )

    def test_live_stop_candidates_keep_broader_results_when_focus_matches_are_too_sparse(self) -> None:
        from functorflow_v3.culinary_tour_agentic import CulinaryTourQueryPlan, _live_stop_candidates

        plan = CulinaryTourQueryPlan(
            query="Plan a kimchi culinary tour in Seoul July 6-11 under $50 per meal",
            normalized_query="plan a kimchi culinary tour in seoul july 6-11 under $50 per meal",
            food_focus="kimchi",
            destination="Seoul",
            time_window="July 6-11",
            budget_per_meal=50,
            estimated_days=6,
        )
        overpass_payload = {
            "elements": [
                {"type": "node", "id": 1, "tags": {"name": "Kimchi House", "cuisine": "kimchi", "price_range": "$$"}},
                {"type": "node", "id": 2, "tags": {"name": "Seoul Bistro", "cuisine": "korean", "price_range": "$"}},
                {"type": "node", "id": 3, "tags": {"name": "Han Table", "cuisine": "korean", "price_range": "$$"}},
            ]
        }
        with patch("functorflow_v3.culinary_tour_agentic._geocode_destination", return_value={"boundingbox": ["37.55", "37.58", "126.97", "127.01"]}):
            with patch("functorflow_v3.culinary_tour_agentic._load_overpass_json", return_value=overpass_payload):
                stops = _live_stop_candidates(plan)

        self.assertEqual(len(stops), 3)

    def test_osm_stop_uses_openstreetmap_link_when_website_missing(self) -> None:
        from functorflow_v3.culinary_tour_agentic import _osm_element_to_stop

        stop = _osm_element_to_stop(
            {
                "type": "node",
                "id": 12345,
                "tags": {
                    "name": "Fallback Kimchi House",
                    "cuisine": "korean",
                },
            },
            destination="Seoul",
        )

        self.assertIsNotNone(stop)
        self.assertEqual(stop.url, "https://www.openstreetmap.org/node/12345")


if __name__ == "__main__":
    unittest.main()
