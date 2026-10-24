"""User-facing inputs: the prompted-mode ``fields.yaml`` and the confirmed ``mapping.json``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ditto.specgen.models import FieldType, TypeCheck

ABSENT = "absent"


class FieldRequest(BaseModel):
    """One requested field. ``fields.yaml`` accepts either form::

    price: the listed price, including currency symbol      # short form
    email: {description: contact email, type: scalar, examples: [a@b.org], type_check: email}
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    type: FieldType | None = None
    type_check: TypeCheck | None = None
    examples: list[str] = Field(default_factory=list)


class FieldsSchema(BaseModel):
    fields: list[FieldRequest]

    @classmethod
    def from_obj(cls, data: Any) -> FieldsSchema:
        if isinstance(data, dict) and isinstance(data.get("fields"), (dict, list)):
            data = data["fields"]
        items: list[FieldRequest] = []
        if isinstance(data, dict):
            for name, spec in data.items():
                if spec is None or isinstance(spec, str):
                    items.append(FieldRequest(name=str(name), description=spec or ""))
                else:
                    items.append(FieldRequest(name=str(name), **spec))
        elif isinstance(data, list):
            items = [
                FieldRequest(**d) if isinstance(d, dict) else FieldRequest(name=str(d))
                for d in data
            ]
        else:
            raise ValueError(
                "fields.yaml must be a mapping of field name -> description or options"
            )
        if not items:
            raise ValueError("fields.yaml defines no fields")
        return cls(fields=items)

    @classmethod
    def load(cls, path: str | Path) -> FieldsSchema:
        return cls.from_obj(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    def get(self, name: str) -> FieldRequest | None:
        return next((f for f in self.fields if f.name == name), None)


class TemplateMapping(BaseModel):
    included: bool = True
    # field name -> candidate id ("c17"), list of candidate ids (record/kv groups), or "absent"
    fields: dict[str, str | list[str]] = Field(default_factory=dict)
    note: str | None = None


class Mapping(BaseModel):
    """``mapping.json``: template -> field -> candidate id(s) or ``absent``."""

    version: Literal[1] = 1
    schema_fields: list[str] = Field(default_factory=list)
    templates: dict[str, TemplateMapping] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Mapping:
        return cls.model_validate_json(Path(path).read_bytes())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2) + "\n")

    def included_templates(self) -> list[str]:
        return [tid for tid, tm in self.templates.items() if tm.included]
