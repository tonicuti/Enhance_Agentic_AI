# Module Spec: `corpus-ingestion`

## Objective
Load `data/corpus.json`, validate article records, normalize metadata, assign stable document IDs, and create traceable text chunks for both retrievers.

## Interface
Input records require `title`, `url`, and `body`. Output documents/chunks retain document ID, title, URL, source, category, publication date, and chunk index. IDs derive from canonical URLs.

## Success Criteria
- Valid records produce deterministic IDs and chunks.
- Required metadata survives splitting unchanged.
- Invalid records are reported with identifiers and do not silently disappear.

## Boundaries
- Always preserve URL provenance.
- Ask first before changing canonicalization or malformed-record policy after indexing begins.
- Never use evaluation `evidence_list` to modify corpus records.

## Implementation choices
The implementation uses Python 3.12+, canonicalizes absolute HTTP(S) URLs by normalizing host/default ports, sorting query parameters, removing fragments and trailing path slashes, and derives stable SHA-256 document and chunk IDs. LangChain `RecursiveCharacterTextSplitter` uses 1,000-character chunks with 150-character overlap by default; all settings are configurable in `configs/default.json`.
