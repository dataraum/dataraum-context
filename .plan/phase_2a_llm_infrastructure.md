# Phase 2A: LLM Infrastructure Implementation Plan

**Status**: Planning  
**Dependencies**: Phase 1 (Storage), Phase 2B (Data Pipeline) - COMPLETED  
**Focus**: Generic LLM framework with Anthropic implementation first  

---

## Overview

This phase builds the LLM infrastructure to power semantic analysis, quality rule generation, suggested queries, and context summaries. The design prioritizes:

1. **Provider abstraction** - Easy to swap/add LLM providers
2. **Anthropic first** - Implement Claude as primary provider
3. **Caching** - Avoid redundant API calls via `llm_cache` table
4. **Privacy awareness** - Prepare for SDV integration (implement later)
5. **Configuration-driven** - All controlled via `config/llm.yaml`

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                  LLM Module Structure                        │
│                                                              │
│  llm/                                                        │
│  ├── config.py          # LLM configuration loader          │
│  ├── providers/         # LLM provider implementations      │
│  │   ├── base.py        # Abstract provider interface       │
│  │   ├── anthropic.py   # Claude implementation [PHASE 2A]  │
│  │   ├── openai.py      # OpenAI stub [LATER]              │
│  │   └── local.py       # Local LLM stub [LATER]           │
│  ├── cache.py           # Response caching logic            │
│  ├── prompts.py         # Prompt template loading/rendering │
│  ├── privacy.py         # Data sampling/SDV placeholder     │
│  └── features/          # LLM-powered features              │
│      ├── semantic.py    # Semantic analysis                 │
│      ├── quality.py     # Quality rule generation           │
│      ├── queries.py     # Suggested query generation        │
│      └── summary.py     # Context summary generation        │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 3: LLM Providers (Generic + Anthropic)

**Goal**: Abstract provider interface + working Anthropic implementation

#### 3.1: Provider Base Interface

**File**: `src/dataraum_context/llm/providers/base.py`

**Models**:
```python
class LLMRequest(BaseModel):
    """Request to LLM provider."""
    prompt: str
    max_tokens: int = 4000
    temperature: float = 0.0
    response_format: str = "json"  # "json" or "text"

class LLMResponse(BaseModel):
    """Response from LLM provider."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool = False  # True if from our cache
    provider_cached: bool = False  # True if provider cache hit

class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    async def complete(self, request: LLMRequest) -> Result[LLMResponse]:
        """Send completion request to provider."""
        pass
    
    @abstractmethod
    def get_model_for_tier(self, tier: str) -> str:
        """Get model name for tier (fast/balanced)."""
        pass
```

**Key Points**:
- Use `Result[LLMResponse]` for error handling
- Support both JSON and text responses
- Track token usage for cost monitoring
- Distinguish between our cache and provider cache

#### 3.2: Anthropic Provider Implementation

**File**: `src/dataraum_context/llm/providers/anthropic.py`

**Implementation**:
```python
class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing env var: {config.api_key_env}")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def complete(self, request: LLMRequest) -> Result[LLMResponse]:
        """Send request to Claude API."""
        try:
            model = self.config.default_model
            
            # Handle JSON mode
            if request.response_format == "json":
                # Use system prompt to enforce JSON
                system = "Respond with valid JSON only. No markdown formatting."
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system=system,
                    messages=[{"role": "user", "content": request.prompt}]
                )
            else:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    messages=[{"role": "user", "content": request.prompt}]
                )
            
            # Extract text content
            content = response.content[0].text
            
            return Result.ok(LLMResponse(
                content=content,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached=False,
                provider_cached=False,  # Anthropic doesn't expose cache hits yet
            ))
            
        except anthropic.APIError as e:
            return Result.fail(f"Anthropic API error: {e}")
        except Exception as e:
            return Result.fail(f"Unexpected error: {e}")
    
    def get_model_for_tier(self, tier: str) -> str:
        """Get model name for tier."""
        return self.config.models.get(tier, self.config.default_model)
```

**Key Points**:
- Use async Anthropic client
- Handle JSON mode via system prompt (Claude doesn't have native JSON mode)
- Proper error handling with Result type
- Track token usage from response

**Dependencies**:
- Add `anthropic>=0.40.0` to optional dependencies ✓ (already in pyproject.toml)

#### 3.3: OpenAI Provider Stub

**File**: `src/dataraum_context/llm/providers/openai.py`

**Implementation**:
```python
class OpenAIProvider(LLMProvider):
    """OpenAI provider stub - implement later."""
    
    def __init__(self, config: ProviderConfig):
        raise NotImplementedError(
            "OpenAI provider not yet implemented. "
            "Use 'anthropic' provider or contribute OpenAI support!"
        )
    
    async def complete(self, request: LLMRequest) -> Result[LLMResponse]:
        raise NotImplementedError()
    
    def get_model_for_tier(self, tier: str) -> str:
        raise NotImplementedError()
```

**Key Points**:
- Stub only - fail fast with clear message
- Implementation can be added later without changing interfaces

#### 3.4: Local Provider Stub

**File**: `src/dataraum_context/llm/providers/local.py`

Same as OpenAI - stub for now.

#### 3.5: Provider Factory

**File**: `src/dataraum_context/llm/providers/__init__.py`

```python
def create_provider(config: LLMConfig) -> LLMProvider:
    """Create LLM provider based on configuration."""
    provider_name = config.active_provider
    
    if provider_name not in config.providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    provider_config = config.providers[provider_name]
    
    if provider_name == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(provider_config)
    elif provider_name == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(provider_config)
    elif provider_name == "local":
        from .local import LocalProvider
        return LocalProvider(provider_config)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")
```

---

### Step 4: LLM Prompts + Privacy (SDV Placeholder)

**Goal**: Prompt template system + data sampling with SDV placeholder

#### 4.1: Configuration Models

**File**: `src/dataraum_context/llm/config.py`

```python
class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    api_key_env: str
    default_model: str
    models: dict[str, str]  # tier -> model name
    base_url_env: str | None = None  # For local providers

class FeatureConfig(BaseModel):
    """Configuration for an LLM feature."""
    enabled: bool = True
    model_tier: str = "balanced"
    prompt_file: str
    description: str = ""

class LLMFeatures(BaseModel):
    """All LLM features."""
    semantic_analysis: FeatureConfig
    quality_rule_generation: FeatureConfig
    suggested_queries: FeatureConfig
    context_summary: FeatureConfig

class LLMLimits(BaseModel):
    """Cost and rate controls."""
    max_input_tokens_per_request: int = 8000
    max_output_tokens_per_request: int = 4000
    max_columns_per_batch: int = 30
    max_requests_per_minute: int = 20
    cache_ttl_seconds: int = 86400  # 24 hours

class LLMPrivacy(BaseModel):
    """Privacy settings."""
    max_sample_values: int = 10
    use_synthetic_samples: bool = False  # SDV integration
    synthetic_sample_count: int = 20
    sensitive_patterns: list[str] = Field(default_factory=list)

class LLMConfig(BaseModel):
    """Complete LLM configuration from llm.yaml."""
    providers: dict[str, ProviderConfig]
    active_provider: str
    features: LLMFeatures
    limits: LLMLimits
    privacy: LLMPrivacy
    fallback: dict[str, str]  # feature -> fallback strategy

def load_llm_config(config_path: Path = Path("config/llm.yaml")) -> LLMConfig:
    """Load LLM configuration from YAML."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return LLMConfig(**data)
```

**Key Points**:
- Pydantic models match `config/llm.yaml` structure
- Clear separation: providers, features, limits, privacy
- Use environment variables for API keys (never in config files)

#### 4.2: Prompt Template System

**File**: `src/dataraum_context/llm/prompts.py`

```python
class PromptTemplate(BaseModel):
    """A prompt template from YAML."""
    name: str
    version: str
    description: str
    temperature: float
    prompt: str  # Template with {variable} placeholders
    inputs: dict[str, Any]
    output_schema: dict[str, Any]
    validation: dict[str, list[str]]

class PromptRenderer:
    """Render prompt templates with context."""
    
    def __init__(self, prompts_dir: Path = Path("config/prompts")):
        self.prompts_dir = prompts_dir
        self._cache: dict[str, PromptTemplate] = {}
    
    def load_template(self, name: str) -> PromptTemplate:
        """Load and cache a prompt template."""
        if name in self._cache:
            return self._cache[name]
        
        template_path = self.prompts_dir / f"{name}.yaml"
        with open(template_path) as f:
            data = yaml.safe_load(f)
        
        template = PromptTemplate(**data)
        self._cache[name] = template
        return template
    
    def render(
        self, 
        template_name: str, 
        context: dict[str, Any]
    ) -> tuple[str, float]:
        """Render a prompt with context variables.
        
        Returns:
            (rendered_prompt, temperature)
        """
        template = self.load_template(template_name)
        
        # Validate required inputs
        for input_name, input_spec in template.inputs.items():
            if input_spec.get("required", False) and input_name not in context:
                raise ValueError(f"Missing required input: {input_name}")
        
        # Fill in defaults
        full_context = {}
        for input_name, input_spec in template.inputs.items():
            if input_name in context:
                full_context[input_name] = context[input_name]
            elif "default" in input_spec:
                full_context[input_name] = input_spec["default"]
        
        # Render template
        rendered = template.prompt.format(**full_context)
        
        return rendered, template.temperature
```

**Key Points**:
- Load prompt templates from `config/prompts/*.yaml`
- Simple string formatting (can upgrade to Jinja2 if needed)
- Validate required inputs and provide defaults
- Cache templates in memory

#### 4.3: Response Cache

**File**: `src/dataraum_context/llm/cache.py`

```python
import hashlib
import json
from datetime import datetime, timedelta, UTC

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.storage.models import LLMCache as LLMCacheModel

class LLMCache:
    """Cache LLM responses to avoid redundant API calls."""
    
    @staticmethod
    def _compute_cache_key(
        feature: str,
        prompt: str,
        model: str,
        table_ids: list[str] | None = None,
    ) -> str:
        """Compute cache key from inputs."""
        # Hash the prompt + model + table IDs
        key_data = {
            "feature": feature,
            "prompt": prompt,
            "model": model,
            "table_ids": sorted(table_ids or []),
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()
    
    async def get(
        self,
        session: AsyncSession,
        feature: str,
        prompt: str,
        model: str,
        table_ids: list[str] | None = None,
    ) -> LLMResponse | None:
        """Get cached response if available and valid."""
        cache_key = self._compute_cache_key(feature, prompt, model, table_ids)
        
        stmt = select(LLMCacheModel).where(
            LLMCacheModel.cache_key == cache_key,
            LLMCacheModel.is_valid == True,
            LLMCacheModel.expires_at > datetime.now(UTC),
        )
        
        result = await session.execute(stmt)
        cache_entry = result.scalar_one_or_none()
        
        if cache_entry:
            return LLMResponse(
                content=cache_entry.response_json["content"],
                model=cache_entry.model,
                input_tokens=cache_entry.input_tokens or 0,
                output_tokens=cache_entry.output_tokens or 0,
                cached=True,
            )
        
        return None
    
    async def put(
        self,
        session: AsyncSession,
        feature: str,
        prompt: str,
        response: LLMResponse,
        source_id: str | None = None,
        table_ids: list[str] | None = None,
        ontology: str | None = None,
        ttl_seconds: int = 86400,
    ) -> None:
        """Store response in cache."""
        cache_key = self._compute_cache_key(feature, prompt, response.model, table_ids)
        
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        
        cache_entry = LLMCacheModel(
            cache_key=cache_key,
            feature=feature,
            source_id=source_id,
            table_ids={"ids": table_ids or []},
            ontology=ontology,
            provider=response.model.split("-")[0],  # Extract provider from model
            model=response.model,
            response_json={
                "content": response.content,
            },
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            expires_at=expires_at,
        )
        
        session.add(cache_entry)
        await session.commit()
    
    async def invalidate_for_source(
        self,
        session: AsyncSession,
        source_id: str,
    ) -> None:
        """Invalidate all cache entries for a source."""
        stmt = (
            select(LLMCacheModel)
            .where(LLMCacheModel.source_id == source_id)
        )
        result = await session.execute(stmt)
        entries = result.scalars().all()
        
        for entry in entries:
            entry.is_valid = False
        
        await session.commit()
```

**Key Points**:
- Cache key = hash(feature + prompt + model + table_ids)
- Check expiry + validity before returning cached response
- Invalidate cache when source data changes
- Store in `llm_cache` table (already defined in storage models ✓)

#### 4.4: Privacy / Data Sampling

**File**: `src/dataraum_context/llm/privacy.py`

```python
class DataSampler:
    """Sample data for LLM analysis with privacy controls."""
    
    def __init__(self, config: LLMPrivacy):
        self.config = config
    
    async def prepare_samples(
        self,
        column_profiles: list[ColumnProfile],
        duckdb_conn: duckdb.DuckDBPyConnection,
    ) -> dict[str, list[Any]]:
        """Prepare sample values for LLM analysis.
        
        Returns:
            dict mapping column_name -> sample_values
        """
        samples = {}
        
        for profile in column_profiles:
            column_name = profile.column_ref.column_name
            
            # Check if column is sensitive
            is_sensitive = any(
                re.match(pattern, column_name, re.IGNORECASE)
                for pattern in self.config.sensitive_patterns
            )
            
            if is_sensitive and self.config.use_synthetic_samples:
                # Placeholder: Use SDV to generate synthetic samples
                # For now, just skip sensitive columns
                samples[column_name] = ["<REDACTED>"] * min(3, self.config.max_sample_values)
            else:
                # Use real top values from profile
                if profile.top_values:
                    samples[column_name] = [
                        vc.value 
                        for vc in profile.top_values[:self.config.max_sample_values]
                    ]
                else:
                    samples[column_name] = []
        
        return samples

# SDV Integration (future)
# When implementing SDV:
# 1. Create separate service/container for SDV (heavy PyTorch dependency)
# 2. Call via HTTP API or message queue
# 3. Generate synthetic samples matching statistical properties
# 4. Use synthesized data instead of real values for LLM prompts
```

**Key Points**:
- Pattern-based sensitive column detection
- SDV placeholder - just redact for now
- Real implementation should be a separate service (PyTorch dependency)
- Return dict of column -> sample values for inclusion in prompts

**SDV Service Design** (for later):
```
┌─────────────────┐         HTTP API        ┌─────────────────┐
│  DataRaum       │  ────────────────────▶  │  SDV Service    │
│  Context Engine │                         │  (separate)     │
│                 │  ◀────────────────────  │                 │
└─────────────────┘    Synthetic samples    └─────────────────┘
                                            
- Separate Docker container
- REST API: POST /synthesize with table profile
- Returns: Synthetic samples matching distribution
- Stateless - no persistence needed
```

---

### Step 5-8: LLM Features (Semantic, Quality, Queries, Summary)

Now we implement the four LLM-powered features. Each follows the same pattern:

1. Load prompt template
2. Prepare context (table profiles + samples)
3. Check cache
4. Call LLM provider
5. Parse response
6. Store in cache
7. Save to metadata tables

#### Common Pattern

**File**: `src/dataraum_context/llm/features/_base.py`

```python
class LLMFeature:
    """Base class for LLM features."""
    
    def __init__(
        self,
        config: LLMConfig,
        provider: LLMProvider,
        prompt_renderer: PromptRenderer,
        cache: LLMCache,
    ):
        self.config = config
        self.provider = provider
        self.renderer = renderer
        self.cache = cache
    
    async def _call_llm(
        self,
        session: AsyncSession,
        feature_name: str,
        prompt: str,
        temperature: float,
        model_tier: str,
        source_id: str | None = None,
        table_ids: list[str] | None = None,
    ) -> Result[LLMResponse]:
        """Call LLM with caching."""
        
        # Get model for tier
        model = self.provider.get_model_for_tier(model_tier)
        
        # Check cache
        cached = await self.cache.get(
            session, feature_name, prompt, model, table_ids
        )
        if cached:
            return Result.ok(cached)
        
        # Call provider
        request = LLMRequest(
            prompt=prompt,
            max_tokens=self.config.limits.max_output_tokens_per_request,
            temperature=temperature,
            response_format="json",
        )
        
        result = await self.provider.complete(request)
        if not result.success:
            return result
        
        # Store in cache
        await self.cache.put(
            session,
            feature_name,
            prompt,
            result.value,
            source_id,
            table_ids,
            ttl_seconds=self.config.limits.cache_ttl_seconds,
        )
        
        return result
```

#### 5.1: Semantic Analysis Feature

**File**: `src/dataraum_context/llm/features/semantic.py`

```python
class SemanticAnalysisFeature(LLMFeature):
    """LLM-powered semantic analysis."""
    
    async def analyze(
        self,
        session: AsyncSession,
        duckdb_conn: duckdb.DuckDBPyConnection,
        table_ids: list[str],
        ontology: str = "general",
    ) -> Result[SemanticEnrichmentResult]:
        """Analyze semantic meaning of tables and columns."""
        
        feature_config = self.config.features.semantic_analysis
        if not feature_config.enabled:
            return Result.fail("Semantic analysis is disabled in config")
        
        # Load column profiles from metadata
        profiles = await self._load_profiles(session, table_ids)
        
        # Prepare sample data
        sampler = DataSampler(self.config.privacy)
        samples = await sampler.prepare_samples(profiles, duckdb_conn)
        
        # Build context for prompt
        tables_json = self._build_tables_json(profiles, samples)
        ontology_data = await self._load_ontology(session, ontology)
        
        context = {
            "tables_json": json.dumps(tables_json, indent=2),
            "ontology_name": ontology,
            "ontology_concepts": self._format_ontology_concepts(ontology_data),
        }
        
        # Render prompt
        prompt, temperature = self.renderer.render(
            feature_config.prompt_file.replace("prompts/", "").replace(".yaml", ""),
            context,
        )
        
        # Call LLM
        response_result = await self._call_llm(
            session,
            "semantic_analysis",
            prompt,
            temperature,
            feature_config.model_tier,
            table_ids=table_ids,
        )
        
        if not response_result.success:
            return Result.fail(response_result.error)
        
        response = response_result.value
        
        # Parse response
        try:
            parsed = json.loads(response.content)
            return self._parse_semantic_response(parsed, table_ids)
        except json.JSONDecodeError as e:
            return Result.fail(f"Failed to parse LLM response: {e}")
    
    def _build_tables_json(
        self, 
        profiles: list[ColumnProfile],
        samples: dict[str, list[Any]],
    ) -> list[dict]:
        """Build JSON representation of tables for prompt."""
        # Group by table
        tables_data = {}
        for profile in profiles:
            table_name = profile.column_ref.table_name
            if table_name not in tables_data:
                tables_data[table_name] = {
                    "table_name": table_name,
                    "columns": []
                }
            
            col_data = {
                "column_name": profile.column_ref.column_name,
                "data_type": "VARCHAR",  # Raw type before resolution
                "null_ratio": profile.null_ratio,
                "distinct_count": profile.distinct_count,
                "sample_values": samples.get(profile.column_ref.column_name, []),
            }
            
            # Add numeric stats if available
            if profile.numeric_stats:
                col_data["min"] = profile.numeric_stats.min_value
                col_data["max"] = profile.numeric_stats.max_value
                col_data["mean"] = profile.numeric_stats.mean
            
            # Add detected patterns
            if profile.detected_patterns:
                col_data["patterns"] = [p.name for p in profile.detected_patterns]
            
            tables_data[table_name]["columns"].append(col_data)
        
        return list(tables_data.values())
    
    def _parse_semantic_response(
        self, 
        parsed: dict,
        table_ids: list[str],
    ) -> Result[SemanticEnrichmentResult]:
        """Parse LLM response into structured result."""
        
        annotations = []
        entity_detections = []
        relationships = []
        
        # Parse table-level entities
        for table_data in parsed.get("tables", []):
            # Create entity detection
            entity = EntityDetection(
                table_id="",  # Will be filled by caller
                table_name=table_data["table_name"],
                entity_type=table_data.get("entity_type", "unknown"),
                description=table_data.get("description"),
                confidence=0.8,  # Default confidence
                evidence={},
                grain_columns=table_data.get("grain", []),
                is_fact_table=table_data.get("is_fact_table", False),
                is_dimension_table=not table_data.get("is_fact_table", False),
                time_column=table_data.get("time_column"),
            )
            entity_detections.append(entity)
            
            # Parse column annotations
            for col_data in table_data.get("columns", []):
                annotation = SemanticAnnotation(
                    column_id="",  # Will be filled by caller
                    column_ref=ColumnRef(
                        table_name=table_data["table_name"],
                        column_name=col_data["column_name"],
                    ),
                    semantic_role=SemanticRole(col_data.get("semantic_role", "unknown")),
                    entity_type=col_data.get("entity_type"),
                    business_name=col_data.get("business_term"),
                    business_description=col_data.get("description"),
                    annotation_source=DecisionSource.LLM,
                    annotated_by=parsed.get("_model", "llm"),
                    confidence=col_data.get("confidence", 0.8),
                )
                annotations.append(annotation)
        
        # Parse relationships
        for rel_data in parsed.get("relationships", []):
            relationship = Relationship(
                relationship_id="",  # Generated later
                from_table=rel_data["from_table"],
                from_column=rel_data["from_column"],
                to_table=rel_data["to_table"],
                to_column=rel_data["to_column"],
                relationship_type=RelationshipType(rel_data.get("relationship_type", "foreign_key")),
                cardinality=Cardinality(rel_data.get("cardinality", "many_to_one")),
                confidence=rel_data.get("confidence", 0.8),
                detection_method="llm",
                evidence={"source": "semantic_analysis"},
            )
            relationships.append(relationship)
        
        return Result.ok(SemanticEnrichmentResult(
            annotations=annotations,
            entity_detections=entity_detections,
            relationships=relationships,
            source="llm",
        ))
```

**Key Points**:
- Load column profiles from metadata
- Sample data with privacy controls
- Build JSON context matching prompt expectations
- Parse JSON response into structured models
- Return `SemanticEnrichmentResult` (defined in `core.models`)

#### 5.2-5.4: Other Features

The remaining features follow the same pattern:

**Quality Rules** (`features/quality.py`):
- Input: Semantic analysis result + ontology
- Output: List of `QualityRule` objects
- Prompt: `config/prompts/quality_rules.yaml`

**Suggested Queries** (`features/queries.py`):
- Input: Semantic analysis result + ontology metrics
- Output: List of `SuggestedQuery` objects
- Prompt: `config/prompts/suggested_queries.yaml`

**Context Summary** (`features/summary.py`):
- Input: Semantic analysis result + quality scores
- Output: `ContextSummary` object
- Prompt: `config/prompts/context_summary.yaml`

---

## Testing Strategy

### Unit Tests

**Test Provider Interface**:
```python
# tests/llm/test_providers.py

@pytest.mark.asyncio
async def test_anthropic_provider_success(mocker):
    """Test successful Anthropic API call."""
    mock_response = Mock()
    mock_response.content = [Mock(text='{"result": "test"}')]
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    
    mocker.patch.object(
        anthropic.AsyncAnthropic,
        "messages.create",
        return_value=mock_response
    )
    
    config = ProviderConfig(
        api_key_env="TEST_KEY",
        default_model="claude-sonnet-4-20250514",
        models={"fast": "claude-haiku-4-20250414"}
    )
    
    provider = AnthropicProvider(config)
    request = LLMRequest(prompt="test", max_tokens=100)
    
    result = await provider.complete(request)
    
    assert result.success
    assert result.value.content == '{"result": "test"}'
    assert result.value.input_tokens == 100
```

**Test Cache**:
```python
# tests/llm/test_cache.py

@pytest.mark.asyncio
async def test_cache_hit(db_session):
    """Test cache retrieval."""
    cache = LLMCache()
    
    # Store response
    response = LLMResponse(
        content="test",
        model="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=50,
    )
    
    await cache.put(
        db_session,
        "test_feature",
        "test prompt",
        response,
        ttl_seconds=3600,
    )
    
    # Retrieve
    cached = await cache.get(
        db_session,
        "test_feature",
        "test prompt",
        "claude-sonnet-4-20250514",
    )
    
    assert cached is not None
    assert cached.content == "test"
    assert cached.cached is True
```

**Test Prompt Rendering**:
```python
# tests/llm/test_prompts.py

def test_prompt_render_with_defaults():
    """Test prompt rendering with default values."""
    renderer = PromptRenderer(prompts_dir=Path("tests/fixtures/prompts"))
    
    prompt, temp = renderer.render("test_prompt", {"required_var": "value"})
    
    assert "value" in prompt
    assert temp == 0.0
```

### Integration Tests

**Test Semantic Analysis End-to-End**:
```python
# tests/llm/test_semantic_integration.py

@pytest.mark.asyncio
async def test_semantic_analysis_integration(
    db_session,
    duckdb_conn,
    sample_profiles,
    mock_anthropic,
):
    """Test full semantic analysis flow."""
    
    # Setup mock LLM response
    mock_anthropic.return_value = {
        "tables": [{
            "table_name": "sales",
            "entity_type": "transaction",
            "description": "Sales transactions",
            "columns": [{
                "column_name": "amount",
                "semantic_role": "measure",
                "entity_type": "monetary_value",
                "business_term": "Sale Amount",
                "description": "Transaction amount in USD",
                "confidence": 0.95
            }]
        }]
    }
    
    # Run analysis
    feature = SemanticAnalysisFeature(config, provider, renderer, cache)
    result = await feature.analyze(
        db_session,
        duckdb_conn,
        table_ids=["test-table-id"],
        ontology="general",
    )
    
    assert result.success
    assert len(result.value.annotations) > 0
    assert result.value.annotations[0].semantic_role == SemanticRole.MEASURE
```

---

## Migration from Existing Code

The existing code already has:
- ✓ Storage models with `LLMCache` table
- ✓ Core models with Result type and data classes
- ✓ Prompt templates in `config/prompts/`
- ✓ LLM config in `config/llm.yaml`

No migration needed - this is net new implementation.

---

## Dependencies & Environment

**Required**:
- `anthropic>=0.40.0` (already in `pyproject.toml` as optional)

**Environment Variables**:
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

**Installation**:
```bash
# Install with Anthropic support
pip install -e ".[anthropic]"

# Or for development
pip install -e ".[dev,anthropic]"
```

---

## Deliverables Checklist

### Step 3: LLM Providers
- [ ] `llm/providers/base.py` - Abstract interface
- [ ] `llm/providers/anthropic.py` - Working Anthropic implementation
- [ ] `llm/providers/openai.py` - Stub with NotImplementedError
- [ ] `llm/providers/local.py` - Stub with NotImplementedError
- [ ] `llm/providers/__init__.py` - Provider factory
- [ ] Tests for provider interface
- [ ] Tests for Anthropic provider (mocked)

### Step 4: Prompts + Privacy
- [ ] `llm/config.py` - Configuration models
- [ ] `llm/prompts.py` - Template loader/renderer
- [ ] `llm/cache.py` - Response caching
- [ ] `llm/privacy.py` - Data sampling with SDV placeholder
- [ ] Tests for config loading
- [ ] Tests for prompt rendering
- [ ] Tests for cache operations
- [ ] Tests for data sampling

### Step 5-8: LLM Features
- [ ] `llm/features/_base.py` - Common feature base class
- [ ] `llm/features/semantic.py` - Semantic analysis
- [ ] `llm/features/quality.py` - Quality rule generation
- [ ] `llm/features/queries.py` - Suggested queries
- [ ] `llm/features/summary.py` - Context summary
- [ ] Integration tests for each feature
- [ ] End-to-end test with real Anthropic API (manual/CI only)

### Documentation
- [ ] Update `docs/INTERFACES.md` with LLM module interfaces
- [ ] Add `docs/LLM_FEATURES.md` explaining each feature
- [ ] Update main `README.md` with LLM setup instructions
- [ ] Add example `.env.example` file

---

## Success Criteria

**Phase 2A is complete when**:

1. ✅ Anthropic provider works and can call Claude API
2. ✅ Prompt templates load and render correctly
3. ✅ Cache stores and retrieves LLM responses
4. ✅ Semantic analysis feature generates annotations
5. ✅ Quality rules feature generates rules
6. ✅ Suggested queries feature generates queries
7. ✅ Context summary feature generates summaries
8. ✅ All tests pass (unit + integration)
9. ✅ Privacy controls prevent sensitive data from being sent to LLM
10. ✅ OpenAI and local providers have stubs for future implementation

**NOT required for Phase 2A**:
- ❌ SDV synthetic data generation (placeholder only)
- ❌ OpenAI provider implementation
- ❌ Local LLM provider implementation
- ❌ Advanced prompt optimization
- ❌ Rate limiting implementation
- ❌ Cost tracking dashboard

---

## Future Enhancements (Post-Phase 2A)

### SDV Service Implementation
- Separate Docker container running SDV
- REST API for synthetic data generation
- Column profile → synthetic samples endpoint
- Statistical validation of synthetic data

### Additional Providers
- OpenAI implementation with native JSON mode
- Local LLM support (Ollama, vLLM)
- Azure OpenAI support
- Custom provider plugin system

### Advanced Features
- Streaming responses for long-running analysis
- Batch processing for large schemas
- Human-in-loop review for LLM outputs
- A/B testing different prompts
- Model performance tracking

---

## Questions for User

Before starting implementation:

1. **API Keys**: Do you have an Anthropic API key set up? (Or should we test with mocks only?)

2. **SDV Priority**: Should we implement SDV integration in Phase 2A, or is the placeholder sufficient for now?

3. **Provider Priority**: Confirm we're doing Anthropic first, OpenAI later?

4. **Test Data**: Do you have sample CSV files we should use for integration testing?

5. **Prompt Tuning**: Should we iterate on the existing prompt templates in `config/prompts/` or use them as-is?

---

## Next Steps

1. Review this plan
2. Answer questions above
3. Start with Step 3.1 (Provider base interface)
4. Implement incrementally following the checklist
5. Test each component before moving to next
6. Integration test with real data at the end

**Estimated Effort**: 3-5 days for full Phase 2A implementation

Let me know if you'd like me to start implementing! 🚀
