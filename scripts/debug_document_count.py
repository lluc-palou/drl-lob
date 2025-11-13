"""
Diagnostic Script: Analyze Document Count Increase

Checks for duplicate documents, timestamps, and hour overlaps to diagnose
why split_X_output has more documents than split_X_input.

Usage:
    python scripts/debug_document_count.py
"""

import os
import sys
from pymongo import MongoClient
from collections import Counter, defaultdict
from datetime import datetime, timedelta

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.utils.logging import logger

# =================================================================================================
# Configuration
# =================================================================================================

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"
SPLIT_ID = 0  # Change this to check different splits

INPUT_COLLECTION = f"split_{SPLIT_ID}_input"
OUTPUT_COLLECTION = f"split_{SPLIT_ID}_output"

# =================================================================================================
# Diagnostic Functions
# =================================================================================================

def check_document_counts(db):
    """Check basic document counts."""
    logger("=" * 80, "INFO")
    logger("DOCUMENT COUNT ANALYSIS", "INFO")
    logger("=" * 80, "INFO")

    input_count = db[INPUT_COLLECTION].count_documents({})
    output_count = db[OUTPUT_COLLECTION].count_documents({})

    logger(f"Input collection:  {input_count:,} documents", "INFO")
    logger(f"Output collection: {output_count:,} documents", "INFO")

    diff = output_count - input_count
    if diff > 0:
        pct = (diff / input_count) * 100
        logger(f"Difference: +{diff:,} documents ({pct:.2f}% increase)", "WARNING")
    elif diff < 0:
        pct = (abs(diff) / input_count) * 100
        logger(f"Difference: {diff:,} documents ({pct:.2f}% decrease)", "WARNING")
    else:
        logger("Difference: 0 documents (counts match!)", "INFO")

    logger("", "INFO")
    return input_count, output_count


def check_duplicate_timestamps(db, collection_name):
    """Check for duplicate timestamps in a collection."""
    logger(f"Checking duplicate timestamps in {collection_name}...", "INFO")

    # Aggregate to find duplicate timestamps
    pipeline = [
        {"$group": {
            "_id": "$timestamp",
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"}
        }},
        {"$match": {"count": {"$gt": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]

    duplicates = list(db[collection_name].aggregate(pipeline))

    if duplicates:
        logger(f"Found {len(duplicates)} duplicate timestamp groups:", "WARNING")
        for dup in duplicates[:5]:
            logger(f"  Timestamp {dup['_id']}: {dup['count']} documents", "WARNING")

        total_duplicates = sum(dup['count'] - 1 for dup in duplicates)
        logger(f"Total extra documents from duplicates: {total_duplicates}", "WARNING")
    else:
        logger(f"No duplicate timestamps found in {collection_name}", "INFO")

    logger("", "INFO")
    return len(duplicates) if duplicates else 0


def check_hourly_distribution(db, collection_name):
    """Check document distribution across hours."""
    logger(f"Checking hourly distribution in {collection_name}...", "INFO")

    pipeline = [
        {"$project": {
            "hour_str": {"$dateToString": {
                "format": "%Y-%m-%dT%H:00:00.000Z",
                "date": "$timestamp"
            }}
        }},
        {"$group": {
            "_id": "$hour_str",
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]

    hourly_counts = list(db[collection_name].aggregate(pipeline))

    if hourly_counts:
        logger(f"Found {len(hourly_counts)} hours:", "INFO")
        logger(f"  First hour: {hourly_counts[0]['_id']} ({hourly_counts[0]['count']:,} docs)", "INFO")
        logger(f"  Last hour:  {hourly_counts[-1]['_id']} ({hourly_counts[-1]['count']:,} docs)", "INFO")

        total_from_hours = sum(h['count'] for h in hourly_counts)
        actual_total = db[collection_name].count_documents({})

        logger(f"  Sum of hourly counts: {total_from_hours:,}", "INFO")
        logger(f"  Actual total count:   {actual_total:,}", "INFO")

        if total_from_hours != actual_total:
            logger(f"  MISMATCH! Difference: {total_from_hours - actual_total:,}", "ERROR")
        else:
            logger(f"  Hourly sum matches total ✓", "INFO")

    logger("", "INFO")
    return hourly_counts


def check_timestamp_boundaries(db, collection_name):
    """Check documents on hour boundaries."""
    logger(f"Checking hour boundary documents in {collection_name}...", "INFO")

    # Find documents with timestamps exactly on the hour (00 minutes, 00 seconds)
    pipeline = [
        {"$project": {
            "timestamp": 1,
            "minute": {"$minute": "$timestamp"},
            "second": {"$second": "$timestamp"},
            "millisecond": {"$millisecond": "$timestamp"}
        }},
        {"$match": {
            "minute": 0,
            "second": 0,
            "millisecond": 0
        }},
        {"$count": "count"}
    ]

    result = list(db[collection_name].aggregate(pipeline))
    boundary_count = result[0]['count'] if result else 0

    logger(f"Documents exactly on hour boundaries: {boundary_count:,}", "INFO")
    logger("", "INFO")
    return boundary_count


def compare_timestamps(db):
    """Compare timestamps between input and output collections."""
    logger("Comparing timestamps between input and output...", "INFO")

    # Get all timestamps from input
    input_timestamps = set(doc['timestamp'] for doc in db[INPUT_COLLECTION].find({}, {'timestamp': 1, '_id': 0}))

    # Get all timestamps from output
    output_timestamps = set(doc['timestamp'] for doc in db[OUTPUT_COLLECTION].find({}, {'timestamp': 1, '_id': 0}))

    only_in_input = input_timestamps - output_timestamps
    only_in_output = output_timestamps - input_timestamps
    in_both = input_timestamps & output_timestamps

    logger(f"Unique timestamps in input:  {len(input_timestamps):,}", "INFO")
    logger(f"Unique timestamps in output: {len(output_timestamps):,}", "INFO")
    logger(f"Timestamps in both:          {len(in_both):,}", "INFO")
    logger(f"Only in input:               {len(only_in_input):,}", "INFO")
    logger(f"Only in output:              {len(only_in_output):,}", "INFO")

    if only_in_output:
        logger("Sample timestamps only in output (first 5):", "WARNING")
        for ts in list(only_in_output)[:5]:
            logger(f"  {ts}", "WARNING")

    logger("", "INFO")
    return len(only_in_output)


def check_hourly_overlaps(db, collection_name):
    """Check if documents appear in multiple hourly windows."""
    logger(f"Checking for hourly window overlaps in {collection_name}...", "INFO")

    # Get hourly groups
    pipeline = [
        {"$project": {
            "timestamp": 1,
            "_id": 1,
            "hour_str": {"$dateToString": {
                "format": "%Y-%m-%dT%H:00:00.000Z",
                "date": "$timestamp"
            }}
        }},
        {"$sort": {"timestamp": 1}}
    ]

    docs = list(db[collection_name].aggregate(pipeline))

    if not docs:
        logger("No documents found", "WARNING")
        return

    # Group by _id and check if any _id appears in multiple hours
    id_to_hours = defaultdict(set)
    for doc in docs:
        id_to_hours[str(doc['_id'])].add(doc['hour_str'])

    multi_hour_docs = {id: hours for id, hours in id_to_hours.items() if len(hours) > 1}

    if multi_hour_docs:
        logger(f"Found {len(multi_hour_docs)} documents in multiple hourly windows:", "ERROR")
        for doc_id, hours in list(multi_hour_docs.items())[:5]:
            logger(f"  ID {doc_id}: {hours}", "ERROR")
    else:
        logger("No documents found in multiple hourly windows ✓", "INFO")

    logger("", "INFO")
    return len(multi_hour_docs)


# =================================================================================================
# Main Function
# =================================================================================================

def main():
    """Run all diagnostic checks."""
    logger("=" * 80, "INFO")
    logger("DOCUMENT COUNT DIAGNOSTIC TOOL", "INFO")
    logger("=" * 80, "INFO")
    logger(f"Database: {DB_NAME}", "INFO")
    logger(f"Split ID: {SPLIT_ID}", "INFO")
    logger("", "INFO")

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    try:
        # Check collections exist
        collections = db.list_collection_names()
        if INPUT_COLLECTION not in collections:
            logger(f"ERROR: {INPUT_COLLECTION} does not exist!", "ERROR")
            return 1
        if OUTPUT_COLLECTION not in collections:
            logger(f"ERROR: {OUTPUT_COLLECTION} does not exist!", "ERROR")
            return 1

        # Run diagnostics
        input_count, output_count = check_document_counts(db)

        if input_count == output_count:
            logger("Counts match - no issue detected!", "INFO")
            return 0

        # Check for duplicates
        logger("=" * 80, "INFO")
        logger("CHECKING FOR DUPLICATE TIMESTAMPS", "INFO")
        logger("=" * 80, "INFO")
        input_dups = check_duplicate_timestamps(db, INPUT_COLLECTION)
        output_dups = check_duplicate_timestamps(db, OUTPUT_COLLECTION)

        # Check hourly distribution
        logger("=" * 80, "INFO")
        logger("HOURLY DISTRIBUTION ANALYSIS", "INFO")
        logger("=" * 80, "INFO")
        input_hours = check_hourly_distribution(db, INPUT_COLLECTION)
        output_hours = check_hourly_distribution(db, OUTPUT_COLLECTION)

        # Check hour boundaries
        logger("=" * 80, "INFO")
        logger("HOUR BOUNDARY ANALYSIS", "INFO")
        logger("=" * 80, "INFO")
        input_boundaries = check_timestamp_boundaries(db, INPUT_COLLECTION)
        output_boundaries = check_timestamp_boundaries(db, OUTPUT_COLLECTION)

        # Compare timestamps
        logger("=" * 80, "INFO")
        logger("TIMESTAMP COMPARISON", "INFO")
        logger("=" * 80, "INFO")
        new_timestamps = compare_timestamps(db)

        # Check for overlaps
        logger("=" * 80, "INFO")
        logger("HOURLY WINDOW OVERLAP CHECK", "INFO")
        logger("=" * 80, "INFO")
        output_overlaps = check_hourly_overlaps(db, OUTPUT_COLLECTION)

        # Summary
        logger("=" * 80, "INFO")
        logger("DIAGNOSTIC SUMMARY", "INFO")
        logger("=" * 80, "INFO")
        logger(f"Document count difference: {output_count - input_count:,}", "INFO")
        logger(f"Output duplicate timestamp groups: {output_dups}", "INFO")
        logger(f"New timestamps in output: {new_timestamps}", "INFO")
        logger(f"Documents in multiple hours: {output_overlaps}", "INFO")
        logger("=" * 80, "INFO")

        return 0

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
