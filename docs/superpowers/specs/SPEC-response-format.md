# Module Spec: `response-format`

## Objective
Return answers in the agreed JSON format with traceable citations.

## Interface
```json
{"role":"assistant","context":"...","sources":["https://example.com/article"]}
```
`role` is `assistant`; `context` contains the grounded answer or an insufficient-evidence notice; `sources` is a list of URLs used to support the answer.

## Success Criteria
- Every output validates against the response contract.
- Every source URL is among the retrieved evidence used for the answer.
- Insufficient evidence produces an empty sources list.

## Boundaries
- Always keep `sources` as URL strings as agreed.
- Ask first before changing the field names or source representation.
- Never fabricate citations.

## Open Decisions
Whether the eventual interaction layer is CLI, API, or UI; response validation library.
