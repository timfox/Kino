"""VeriTrip agent toolset facade over MRB + restaurant APIs (Table 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.veritrip.mrb import MultimodalRetrievalBase
from ltx_trainer.veritrip.vkb import VerifiableKnowledgeBase


@dataclass
class RestaurantAPI:
    """Structured dining metadata (Sec. 3.1); VKB-sourced in demo."""

    by_city: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def recommend(self, city: str) -> list[dict[str, Any]]:
        return list(self.by_city.get(city, []))

    def by_name(self, city: str, name: str) -> dict[str, Any] | None:
        for r in self.by_city.get(city, []):
            if name.lower() in r.get("name", "").lower():
                return r
        return None

    def by_food(self, city: str, food: str) -> list[dict[str, Any]]:
        out = []
        for r in self.by_city.get(city, []):
            cuisine = str(r.get("cuisine", ""))
            foods = str(r.get("recommended_food", ""))
            if food.lower() in cuisine.lower() or food.lower() in foods.lower():
                out.append(r)
        return out


@dataclass
class VeriTripToolset:
    mrb: MultimodalRetrievalBase
    restaurants: RestaurantAPI
    vkb: VerifiableKnowledgeBase | None = None  # not exposed to agents

    def doc_search(self, query: str) -> list[dict[str, Any]]:
        return self.mrb.doc_search(query)

    def img_search(self, img_path: str) -> list[dict[str, Any]]:
        return self.mrb.img_search(img_path)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self.mrb.get_document(doc_id)

    def get_recommend_restaurant(self, city: str) -> list[dict[str, Any]]:
        return self.restaurants.recommend(city)

    def get_restaurant_by_name(self, city: str, name: str) -> dict[str, Any] | None:
        return self.restaurants.by_name(city, name)

    def get_restaurant_by_food(self, city: str, food: str) -> list[dict[str, Any]]:
        return self.restaurants.by_food(city, food)

    def dispatch(self, tool: str, **kwargs: Any) -> Any:
        table = {
            "docSearch": lambda: self.doc_search(str(kwargs.get("query", ""))),
            "imgSearch": lambda: self.img_search(str(kwargs.get("imgPath", ""))),
            "getDocument": lambda: self.get_document(str(kwargs.get("docID", ""))),
            "getRecommendRestaurant": lambda: self.get_recommend_restaurant(str(kwargs.get("city", ""))),
            "getRestaurantByName": lambda: self.get_restaurant_by_name(
                str(kwargs.get("city", "")), str(kwargs.get("name", ""))
            ),
            "getRestaurantByFood": lambda: self.get_restaurant_by_food(
                str(kwargs.get("city", "")), str(kwargs.get("food", ""))
            ),
        }
        if tool not in table:
            raise KeyError(f"unknown tool: {tool}")
        return table[tool]()


def build_demo_toolset() -> VeriTripToolset:
    from ltx_trainer.veritrip.mrb import build_demo_mrb
    from ltx_trainer.veritrip.vkb import build_demo_vkb

    vkb = build_demo_vkb()
    restaurants = RestaurantAPI(
        by_city={
            "Changsha": [
                {
                    "name": "Old Hunan Tea House",
                    "price": 120,
                    "cuisine": "Hunan",
                    "recommended_food": "West Lake Vinegar Fish|Longjing Shrimp",
                    "area": "Kaifu",
                }
            ]
        }
    )
    return VeriTripToolset(mrb=build_demo_mrb(), restaurants=restaurants, vkb=vkb)
