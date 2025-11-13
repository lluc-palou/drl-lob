"""
Quick Duplicate Checker for Split Collections

Checks if a split collection has duplicate timestamps.

Usage:
    python scripts/check_duplicates.py
"""

import os
import sys
from pymongo import MongoClient

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.utils.logging import logger

# Configuration
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "raw"
SPLIT_ID = 0  # Change to check different splits

COLLECTION = f"split_{SPLIT_ID}_input"


def main():
    """Check for duplicate timestamps in collection."""
    logger("=" * 80, "INFO")
    logger("DUPLICATE TIMESTAMP CHECKER", "INFO")
    logger("=" * 80, "INFO")
    logger(f"Collection: {COLLECTION}", "INFO")
    logger("", "INFO")

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    try:
        # Get total count
        total_count = db[COLLECTION].count_documents({})
        logger(f"Total documents: {total_count:,}", "INFO")
        logger("", "INFO")

        # Check for duplicate timestamps
        logger("Checking for duplicate timestamps...", "INFO")

        pipeline = [
            {"$group": {
                "_id": "$timestamp",
                "count": {"$sum": 1}
            }},
            {"$match": {"count": {"$gt": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 20}
        ]

        duplicates = list(db[COLLECTION].aggregate(pipeline))

        if duplicates:
            logger(f"FOUND {len(duplicates)} duplicate timestamp groups!", "ERROR")
            logger("", "INFO")

            total_extra = sum(dup['count'] - 1 for dup in duplicates)
            logger(f"Total duplicate documents: {total_extra:,}", "ERROR")
            logger(f"Percentage of collection: {(total_extra/total_count)*100:.2f}%", "ERROR")
            logger("", "INFO")

            logger("Top duplicates:", "ERROR")
            for i, dup in enumerate(duplicates[:10], 1):
                logger(f"  {i}. Timestamp {dup['_id']}: {dup['count']} copies", "ERROR")

            logger("", "INFO")
            logger("=" * 80, "INFO")
            logger("DIAGNOSIS: Documents with duplicate timestamps found", "ERROR")
            logger("This explains the document count increase!", "ERROR")
            logger("=" * 80, "INFO")
        else:
            logger("No duplicate timestamps found ✓", "INFO")
            logger("", "INFO")
            logger("=" * 80, "INFO")
            logger("DIAGNOSIS: All timestamps are unique", "INFO")
            logger("The count increase must have another cause", "INFO")
            logger("=" * 80, "INFO")

        # Check unique timestamp count
        logger("", "INFO")
        logger("Counting unique timestamps...", "INFO")

        unique_pipeline = [
            {"$group": {"_id": "$timestamp"}},
            {"$count": "unique_count"}
        ]

        unique_result = list(db[COLLECTION].aggregate(unique_pipeline))
        unique_count = unique_result[0]['unique_count'] if unique_result else 0

        logger(f"Unique timestamps: {unique_count:,}", "INFO")
        logger(f"Total documents:   {total_count:,}", "INFO")

        if unique_count < total_count:
            diff = total_count - unique_count
            logger(f"Difference:        {diff:,} duplicate documents", "ERROR")
        else:
            logger("All timestamps are unique ✓", "INFO")

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
