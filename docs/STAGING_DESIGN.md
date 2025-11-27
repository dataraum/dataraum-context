# Staging Module Design

## Overview

The staging module loads data from various sources into DuckDB with a **hybrid approach** based on source type system strength.

## Architecture

### Three Routes Based on Type System Strength

#### Route 1: Untyped Sources (VARCHAR-first)
**Sources:** CSV, JSON, text files

**Flow:**
```
Source → raw_{table} (all VARCHAR) → profiling → typed_{table} + quarantine_{table}
```

**Rationale:** No inherent type information. Need pattern detection to infer types safely.

#### Route 2: Weakly-Typed Sources (VARCHAR-first + type hints)
**Sources:** SQLite, Excel

**Flow:**
```
Source → raw_{table} (all VARCHAR) + type_hints metadata → profiling → typed_{table} + quarantine_{table}
```

**Type hints stored separately:**
```python
TypeHint(column_name="price", suggested_type="DECIMAL", source="sqlite_affinity")
```

**Rationale:** These sources provide type hints but don't enforce them. SQLite's type affinity is advisory. Excel dates are notoriously unreliable.

**Benefits:**
- Preserve original data (prevent silent corruption)
- Use type hints to boost confidence in pattern detection
- Still validate with pattern detection before committing to typed table

#### Route 3: Strongly-Typed Sources (Direct load)
**Sources:** PostgreSQL, MySQL, Parquet, Arrow

**Flow:**
```
Source → typed_{table} (with source types)
       ↓
    profiling (semantic only - no type inference)
```

**Rationale:** These sources enforce types. Trust them and skip type inference entirely.

**Semantic profiling still runs to detect:**
- Patterns (emails, UUIDs, URLs in VARCHAR columns)
- Units (currency, measurements in NUMERIC columns)
- Semantic roles (keys, measures, dimensions)

## Implementation Structure

```
src/dataraum_context/staging/
├── __init__.py
├── base.py              # Base loader ABC, type system classification
├── loaders/
│   ├── __init__.py
│   ├── csv.py           # CSV loader (untyped)
│   ├── json.py          # JSON loader (untyped)
│   ├── sqlite.py        # SQLite loader (weakly-typed)
│   ├── postgres.py      # PostgreSQL loader (strongly-typed)
│   ├── parquet.py       # Parquet loader (strongly-typed)
│   └── arrow.py         # Arrow loader (strongly-typed)
├── registry.py          # Loader registry by source_type
└── coordinator.py       # Orchestrates loading
```

## Data Model Extensions

### Source Metadata
```python
class Source:
    ...
    type_system_strength: str  # 'untyped', 'weak', 'strong'
```

### Type Hints Table
```python
class TypeHint(Base):
    """Type hints from weakly-typed sources."""
    __tablename__ = "type_hints"
    
    hint_id: str (PK)
    column_id: str (FK)
    suggested_type: str
    source_system: str  # 'sqlite_affinity', 'excel_format'
    confidence: float
    created_at: datetime
```

### Table Layer Types
```python
# Route 1 & 2: Untyped/Weakly-typed
"raw"        # VARCHAR-first staging
"typed"      # After type resolution
"quarantine" # Failed casts

# Route 3: Strongly-typed
"typed"      # Direct load from source
# No raw or quarantine layers
```

## Interface: Base Loader

```python
class TypeSystemStrength(str, Enum):
    UNTYPED = "untyped"      # CSV, JSON
    WEAK = "weak"            # SQLite, Excel
    STRONG = "strong"        # PostgreSQL, Parquet

class LoaderBase(ABC):
    """Base class for all data loaders."""
    
    @property
    @abstractmethod
    def type_system_strength(self) -> TypeSystemStrength:
        """Classify the source's type system."""
        pass
    
    @abstractmethod
    async def load(
        self, 
        source_config: SourceConfig,
        duckdb_conn: DuckDBConnection,
    ) -> Result[StagingResult]:
        """Load data into DuckDB."""
        pass
    
    @abstractmethod
    async def get_schema(
        self,
        source_config: SourceConfig,
    ) -> Result[list[ColumnInfo]]:
        """Get source schema information."""
        pass
```

## Loader Implementations

### CSV Loader (Untyped)
```python
class CSVLoader(LoaderBase):
    type_system_strength = TypeSystemStrength.UNTYPED
    
    async def load(self, config, conn):
        # Use DuckDB's CSV reader with all columns as VARCHAR
        sql = f"""
            CREATE TABLE raw_{table_name} AS
            SELECT * FROM read_csv(
                '{path}',
                columns = {{'col1': 'VARCHAR', 'col2': 'VARCHAR', ...}},
                header = true,
                nullstr = {null_values_from_config}
            )
        """
        # No type hints generated
```

### SQLite Loader (Weakly-typed)
```python
class SQLiteLoader(LoaderBase):
    type_system_strength = TypeSystemStrength.WEAK
    
    async def load(self, config, conn):
        # Read SQLite type declarations
        cursor.execute("PRAGMA table_info(table_name)")
        type_hints = [
            TypeHint(
                column_name=row['name'],
                suggested_type=map_sqlite_affinity(row['type']),
                source_system='sqlite_affinity',
                confidence=0.7  # Advisory only
            )
        ]
        
        # Load as VARCHAR
        sql = f"""
            CREATE TABLE raw_{table_name} AS
            SELECT * FROM sqlite_scan('{db_path}', '{table_name}')
            -- Cast all to VARCHAR
        """
        
        # Store type hints in metadata table
        await store_type_hints(session, column_id, type_hints)
```

### PostgreSQL Loader (Strongly-typed)
```python
class PostgresLoader(LoaderBase):
    type_system_strength = TypeSystemStrength.STRONG
    
    async def load(self, config, conn):
        # Read actual PostgreSQL types
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
        """)
        
        # Load with native types - NO VARCHAR conversion
        sql = f"""
            CREATE TABLE typed_{table_name} AS
            SELECT * FROM postgres_scan('{conn_string}', '{table_name}')
        """
        
        # Table goes directly to 'typed' layer
        # No type hints needed - types are enforced
```

### Parquet Loader (Strongly-typed)
```python
class ParquetLoader(LoaderBase):
    type_system_strength = TypeSystemStrength.STRONG
    
    async def load(self, config, conn):
        # Parquet has schema embedded
        # Load with native types
        sql = f"""
            CREATE TABLE typed_{table_name} AS
            SELECT * FROM read_parquet('{path}')
        """
        
        # Direct to typed layer
```

## Coordinator Logic

```python
async def stage_source(
    source_config: SourceConfig,
    duckdb_conn: DuckDBConnection,
    session: AsyncSession,
) -> Result[StagingResult]:
    """Coordinate staging based on source type."""
    
    # Get loader from registry
    loader = get_loader(source_config.source_type)
    
    # Load data
    result = await loader.load(source_config, duckdb_conn)
    
    # Determine next step based on type system strength
    if loader.type_system_strength == TypeSystemStrength.STRONG:
        # Already in typed layer, mark for semantic profiling only
        await mark_for_semantic_profiling(session, result.tables)
    else:
        # In raw layer, needs full profiling + type inference
        await mark_for_type_inference(session, result.tables)
    
    return result
```

## Profiling Module Adjustments

The profiling module needs to know which route was taken:

```python
async def profile_table(
    table: Table,
    duckdb_conn: DuckDBConnection,
) -> Result[ProfileResult]:
    """Profile a table based on its layer."""
    
    if table.layer == "raw":
        # Full profiling: statistical + pattern detection + type inference
        return await profile_with_type_inference(table, duckdb_conn)
    
    elif table.layer == "typed" and table.source.type_system_strength == "strong":
        # Semantic profiling only: patterns + units
        # Skip type inference (trust source types)
        return await profile_semantic_only(table, duckdb_conn)
    
    else:
        raise ValueError(f"Unexpected state: {table.layer}, {table.source.type_system_strength}")
```

## Benefits of This Approach

### Correctness
- Untyped/weakly-typed: No silent data loss from premature type coercion
- Strongly-typed: Trust enforced types, avoid unnecessary conversion

### Performance
- Strongly-typed sources skip VARCHAR conversion and type inference
- Only pay the cost of VARCHAR staging when actually needed

### Edge Case Handling
- SQLite's loose typing handled explicitly as "weakly-typed"
- Excel date ambiguity treated with appropriate skepticism
- Clear separation between "type hints" and "enforced types"

### Explainability
- Type system strength is explicit metadata
- Type hints are auditable (stored in database)
- Clear provenance: auto vs. hint-boosted vs. source-enforced

## Migration Notes

- Add `type_system_strength` column to `sources` table
- Create `type_hints` table
- Update `Table.layer` semantics (strongly-typed sources start at 'typed')

## Open Questions

1. **Parquet with incorrect types**: What if a Parquet file claims a column is INT but it's actually corrupt? Do we validate or trust blindly?
   - **Decision**: Trust for now, handle in quality module

2. **Type hint confidence scoring**: How do we score SQLite affinity vs. Excel format hints?
   - **Decision**: SQLite affinity = 0.7, Excel format = 0.5, PostgreSQL = 1.0

3. **Mixed-strength sources**: What if a single source (e.g., CSV export from PostgreSQL) has comments with type info?
   - **Decision**: Classify by format, not origin. CSV is always untyped.

## Testing Strategy

### Unit Tests
- Each loader with fixture data
- Type hint extraction for weakly-typed sources
- DuckDB SQL generation for each route

### Integration Tests
- End-to-end for each source type
- Verify correct layer assignment
- Verify type hints stored for weakly-typed sources

### Property-Based Tests
- Round-trip property: load → typed → query should match source (for strongly-typed)
- No data loss property: raw VARCHAR should preserve all original values
