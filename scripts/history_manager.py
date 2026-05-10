#!/usr/bin/env python3
"""
History Manager - View, compare, favorite, search, and tag ComfyUI generation history
Stores metadata locally in a JSON database for fast querying
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Default history database location
DEFAULT_DB_PATH = os.path.expanduser("~/.comfyui_history.json")


def load_db(db_path):
    """Load history database from JSON file."""
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"records": [], "favorites": [], "tags": {}}
    return {"records": [], "favorites": [], "tags": {}}


def save_db(db, db_path):
    """Save history database to JSON file."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def next_record_id(db):
    """Return a stable new ID even after records have been deleted."""
    existing_ids = [r.get("id", 0) for r in db.get("records", []) if isinstance(r.get("id"), int)]
    return max(existing_ids, default=0) + 1


def add_record(db, record):
    """Add a generation record to history."""
    entry = {
        "id": record.get("id") or next_record_id(db),
        "timestamp": datetime.now().isoformat(),
        "prompt_id": record.get("prompt_id", ""),
        "model": record.get("model", ""),
        "positive": record.get("positive", ""),
        "negative": record.get("negative", ""),
        "workflow": record.get("workflow", ""),
        "function": record.get("function", ""),
        "ecommerce_scene": record.get("ecommerce_scene", ""),
        "images": record.get("images", []),
        "params": record.get("params", {}),
        "tags": record.get("tags", []),
        "favorite": bool(record.get("favorite", False)),
        "rating": record.get("rating", 0),
        "notes": record.get("notes", ""),
    }
    db["records"].append(entry)
    return entry


def list_records(db, limit=20, function_filter=None, model_filter=None,
                 tag_filter=None, favorites_only=False, search_text=None):
    """List history records with optional filtering."""
    records = db["records"]

    if favorites_only:
        records = [r for r in records if r.get("favorite")]

    if function_filter:
        records = [r for r in records if r.get("function") == function_filter]

    if model_filter:
        records = [r for r in records if model_filter.lower() in r.get("model", "").lower()]

    if tag_filter:
        records = [r for r in records if tag_filter in r.get("tags", [])]

    if search_text:
        search_lower = search_text.lower()
        records = [r for r in records if
                   search_lower in r.get("positive", "").lower() or
                   search_lower in r.get("negative", "").lower() or
                   search_lower in r.get("notes", "").lower()]

    # Sort by timestamp descending (newest first)
    records = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)

    return records[:limit]


def toggle_favorite(db, record_id):
    """Toggle favorite status of a record."""
    for record in db["records"]:
        if record["id"] == record_id:
            record["favorite"] = not record.get("favorite", False)
            return record
    return None


def rate_record(db, record_id, rating):
    """Set rating (1-5) for a record."""
    if not 1 <= rating <= 5:
        print("Rating must be 1-5", file=sys.stderr)
        return None
    for record in db["records"]:
        if record["id"] == record_id:
            record["rating"] = rating
            return record
    return None


def add_tags(db, record_id, tags):
    """Add tags to a record."""
    for record in db["records"]:
        if record["id"] == record_id:
            existing = set(record.get("tags", []))
            existing.update(tags)
            record["tags"] = sorted(existing)
            return record
    return None


def remove_tags(db, record_id, tags):
    """Remove tags from a record."""
    for record in db["records"]:
        if record["id"] == record_id:
            existing = set(record.get("tags", []))
            existing -= set(tags)
            record["tags"] = sorted(existing)
            return record
    return None


def add_notes(db, record_id, notes):
    """Add notes to a record."""
    for record in db["records"]:
        if record["id"] == record_id:
            record["notes"] = notes
            return record
    return None


def get_record(db, record_id):
    """Get a specific record by ID."""
    for record in db["records"]:
        if record["id"] == record_id:
            return record
    return None


def compare_records(db, record_ids):
    """Compare multiple records side by side."""
    records = []
    for rid in record_ids:
        r = get_record(db, rid)
        if r:
            records.append(r)
    return records


def get_stats(db):
    """Get generation statistics."""
    records = db["records"]
    if not records:
        return {"total": 0}

    models = {}
    functions = {}
    for r in records:
        model = r.get("model", "unknown")
        func = r.get("function", "unknown")
        models[model] = models.get(model, 0) + 1
        functions[func] = functions.get(func, 0) + 1

    return {
        "total": len(records),
        "favorites": sum(1 for r in records if r.get("favorite")),
        "avg_rating": round(sum(r.get("rating", 0) for r in records if r.get("rating", 0) > 0) /
                           max(1, sum(1 for r in records if r.get("rating", 0) > 0)), 1),
        "top_models": sorted(models.items(), key=lambda x: x[1], reverse=True)[:5],
        "top_functions": sorted(functions.items(), key=lambda x: x[1], reverse=True)[:5],
        "date_range": {
            "first": min(r.get("timestamp", "") for r in records),
            "last": max(r.get("timestamp", "") for r in records),
        },
    }


def import_from_comfyui(db, comfyui_path):
    """Import history from ComfyUI's output directory."""
    output_dir = Path(comfyui_path) / "output"
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return 0

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    imported = 0

    for img_path in sorted(output_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in image_extensions:
            continue

        # Check if already imported
        existing = any(str(img_path) in str(r.get("images", [])) for r in db["records"])
        if existing:
            continue

        # Extract metadata from filename (ComfyUI uses prefix_seed format)
        name = img_path.stem
        entry = add_record(db, {
            "images": [{"filename": img_path.name, "subfolder": str(img_path.parent.relative_to(output_dir))}],
            "notes": f"Imported from ComfyUI output: {img_path}",
        })
        imported += 1

    return imported


def export_records(db, output_path, format="json"):
    """Export history records to file."""
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(db["records"], f, indent=2, ensure_ascii=False)
    elif format == "csv":
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if not db["records"]:
                return
            writer = csv.DictWriter(f, fieldnames=["id", "timestamp", "model", "positive",
                                                    "negative", "function", "tags", "rating", "favorite"])
            writer.writeheader()
            for r in db["records"]:
                row = {k: r.get(k, "") for k in writer.fieldnames}
                if isinstance(row["tags"], list):
                    row["tags"] = "|".join(row["tags"])
                writer.writerow(row)
    print(f"Exported {len(db['records'])} records to {output_path}")


def format_record_short(record):
    """Format a record for short display."""
    fav = "★" if record.get("favorite") else " "
    rating = f"[{'★' * record.get('rating', 0)}{'☆' * (5 - record.get('rating', 0))}]" if record.get("rating") else ""
    tags = " ".join(f"#{t}" for t in record.get("tags", []))
    ts = record.get("timestamp", "")[:16]
    model = record.get("model", "N/A")
    pos = record.get("positive", "")[:60]
    images = len(record.get("images", []))

    return f"  {fav} #{record['id']} {rating} {ts} | {model} | {pos}... | {images}img {tags}"


def format_record_detail(record):
    """Format a record for detailed display."""
    lines = [
        f"--- Record #{record['id']} ---",
        f"Time: {record.get('timestamp', 'N/A')}",
        f"Model: {record.get('model', 'N/A')}",
        f"Function: {record.get('function', 'N/A')}",
        f"Prompt ID: {record.get('prompt_id', 'N/A')}",
        f"Favorite: {'Yes' if record.get('favorite') else 'No'}",
        f"Rating: {'★' * record.get('rating', 0)}{'☆' * (5 - record.get('rating', 0))}",
        f"Tags: {', '.join(record.get('tags', [])) or 'None'}",
        f"Positive: {record.get('positive', 'N/A')}",
        f"Negative: {record.get('negative', 'N/A')}",
        f"Images: {len(record.get('images', []))}",
    ]
    for img in record.get("images", []):
        lines.append(f"  - {img.get('filename', 'unknown')}")
    if record.get("notes"):
        lines.append(f"Notes: {record['notes']}")
    params = record.get("params", {})
    if params:
        lines.append(f"Params: {json.dumps(params)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Generation History Manager")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="History database path")

    subparsers = parser.add_subparsers(dest="command")

    # list - List history records
    list_parser = subparsers.add_parser("list", help="List history records")
    list_parser.add_argument("--limit", type=int, default=20, help="Max records to show")
    list_parser.add_argument("--function", help="Filter by function type")
    list_parser.add_argument("--model", help="Filter by model name (partial match)")
    list_parser.add_argument("--tag", help="Filter by tag")
    list_parser.add_argument("--favorites", action="store_true", help="Show only favorites")
    list_parser.add_argument("--search", help="Search in prompts and notes")

    # show - Show record details
    show_parser = subparsers.add_parser("show", help="Show record details")
    show_parser.add_argument("id", type=int, help="Record ID")

    # favorite - Toggle favorite
    fav_parser = subparsers.add_parser("favorite", help="Toggle favorite status")
    fav_parser.add_argument("id", type=int, help="Record ID")

    # rate - Rate a record
    rate_parser = subparsers.add_parser("rate", help="Rate a record (1-5 stars)")
    rate_parser.add_argument("id", type=int, help="Record ID")
    rate_parser.add_argument("rating", type=int, help="Rating (1-5)")

    # tag - Add/remove tags
    tag_parser = subparsers.add_parser("tag", help="Add or remove tags")
    tag_parser.add_argument("id", type=int, help="Record ID")
    tag_parser.add_argument("--add", nargs="+", help="Tags to add")
    tag_parser.add_argument("--remove", nargs="+", help="Tags to remove")

    # notes - Add notes
    notes_parser = subparsers.add_parser("notes", help="Add notes to a record")
    notes_parser.add_argument("id", type=int, help="Record ID")
    notes_parser.add_argument("text", help="Notes text")

    # compare - Compare records
    compare_parser = subparsers.add_parser("compare", help="Compare multiple records")
    compare_parser.add_argument("ids", nargs="+", type=int, help="Record IDs to compare")

    # stats - Show statistics
    subparsers.add_parser("stats", help="Show generation statistics")

    # import - Import from ComfyUI output
    import_parser = subparsers.add_parser("import", help="Import from ComfyUI output directory")
    import_parser.add_argument("--comfyui-path", required=True, help="ComfyUI path")

    # export - Export records
    export_parser = subparsers.add_parser("export", help="Export history records")
    export_parser.add_argument("--output", required=True, help="Output file path")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json")

    # add - Manually add a record
    add_parser = subparsers.add_parser("add", help="Manually add a generation record")
    add_parser.add_argument("--model", required=True, help="Model name")
    add_parser.add_argument("--positive", required=True, help="Positive prompt")
    add_parser.add_argument("--negative", default="", help="Negative prompt")
    add_parser.add_argument("--function", default="", help="Function type")
    add_parser.add_argument("--prompt-id", default="", help="ComfyUI prompt ID")
    add_parser.add_argument("--images", nargs="+", help="Image filenames")
    add_parser.add_argument("--tags", nargs="+", help="Tags")

    # delete - Delete a record
    del_parser = subparsers.add_parser("delete", help="Delete a record")
    del_parser.add_argument("id", type=int, help="Record ID")

    args = parser.parse_args()
    db = load_db(args.db)

    if args.command == "list":
        records = list_records(db, args.limit, args.function, args.model,
                               args.tag, args.favorites, args.search)
        if not records:
            print("No records found.")
        else:
            print(f"History Records ({len(records)} shown):")
            print("-" * 80)
            for r in records:
                print(format_record_short(r))

    elif args.command == "show":
        record = get_record(db, args.id)
        if record:
            print(format_record_detail(record))
        else:
            print(f"Record #{args.id} not found.")
            sys.exit(1)

    elif args.command == "favorite":
        record = toggle_favorite(db, args.id)
        if record:
            status = "added to" if record["favorite"] else "removed from"
            print(f"Record #{args.id} {status} favorites.")
            save_db(db, args.db)
        else:
            print(f"Record #{args.id} not found.")
            sys.exit(1)

    elif args.command == "rate":
        record = rate_record(db, args.id, args.rating)
        if record:
            print(f"Record #{args.id} rated {'★' * args.rating}")
            save_db(db, args.db)
        else:
            print(f"Record #{args.id} not found.")
            sys.exit(1)

    elif args.command == "tag":
        if args.add:
            record = add_tags(db, args.id, args.add)
            if record:
                print(f"Added tags {args.add} to record #{args.id}")
                save_db(db, args.db)
            else:
                print(f"Record #{args.id} not found.")
                sys.exit(1)
        if args.remove:
            record = remove_tags(db, args.id, args.remove)
            if record:
                print(f"Removed tags {args.remove} from record #{args.id}")
                save_db(db, args.db)
            else:
                print(f"Record #{args.id} not found.")
                sys.exit(1)

    elif args.command == "notes":
        record = add_notes(db, args.id, args.text)
        if record:
            print(f"Notes updated for record #{args.id}")
            save_db(db, args.db)
        else:
            print(f"Record #{args.id} not found.")
            sys.exit(1)

    elif args.command == "compare":
        records = compare_records(db, args.ids)
        if not records:
            print("No matching records found.")
            sys.exit(1)
        print(f"Comparing {len(records)} records:")
        print("=" * 80)
        for r in records:
            print(format_record_detail(r))
            print()

    elif args.command == "stats":
        stats = get_stats(db)
        print("Generation Statistics:")
        print("-" * 40)
        print(f"Total records: {stats.get('total', 0)}")
        print(f"Favorites: {stats.get('favorites', 0)}")
        print(f"Average rating: {stats.get('avg_rating', 0)}")
        if stats.get("top_models"):
            print(f"\nTop models:")
            for model, count in stats["top_models"]:
                print(f"  {model}: {count} times")
        if stats.get("top_functions"):
            print(f"\nTop functions:")
            for func, count in stats["top_functions"]:
                print(f"  {func}: {count} times")
        if stats.get("date_range"):
            print(f"\nDate range: {stats['date_range']['first'][:10]} to {stats['date_range']['last'][:10]}")

    elif args.command == "import":
        count = import_from_comfyui(db, args.comfyui_path)
        if count > 0:
            save_db(db, args.db)
            print(f"Imported {count} new records.")
        else:
            print("No new records to import.")

    elif args.command == "export":
        export_records(db, args.output, args.format)

    elif args.command == "add":
        entry = add_record(db, {
            "model": args.model,
            "positive": args.positive,
            "negative": args.negative,
            "function": args.function,
            "prompt_id": args.prompt_id,
            "images": [{"filename": f} for f in (args.images or [])],
            "tags": args.tags or [],
        })
        save_db(db, args.db)
        print(f"Added record #{entry['id']}")

    elif args.command == "delete":
        before = len(db["records"])
        db["records"] = [r for r in db["records"] if r["id"] != args.id]
        if len(db["records"]) < before:
            save_db(db, args.db)
            print(f"Deleted record #{args.id}")
        else:
            print(f"Record #{args.id} not found.")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
