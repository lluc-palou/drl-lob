"""
Diagnostic script to check what transformations and standardizations were applied to features.
"""

from pymongo import MongoClient
import json

def main():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['raw']

    print("=" * 100)
    print("STAGE 7: FEATURE TRANSFORMATION - What was applied?")
    print("=" * 100)

    # Get transformation config
    transform_config = db['feature_transformation_config'].find_one({})
    if transform_config:
        selected = transform_config.get('selected_transformations', {})
        transform_params = transform_config.get('transformation_parameters', {})

        # Check if volatility was transformed (BUG!)
        if 'volatility' in selected:
            print(f"\n❌ BUG DETECTED: VOLATILITY WAS TRANSFORMED!")
            print(f"   Transformation: {selected['volatility']}")
            if 'volatility' in transform_params:
                print(f"   Parameters: {json.dumps(transform_params['volatility'], indent=6)}")
        else:
            print("\n✓ volatility: Correctly excluded from transformation")

        # Show all transformations
        print(f"\nTotal features transformed: {len(selected)}")
        print("\nTransformation Summary:")
        print(f"{'Feature':<30} {'Transform':<15} {'Details'}")
        print("-" * 100)

        for feat in sorted(selected.keys()):
            transform = selected[feat]
            params = transform_params.get(feat, {})

            # Format parameters
            if transform == 'log' and 'offset' in params:
                details = f"offset={params['offset']:.6f}"
            elif transform == 'sqrt' and 'offset' in params:
                details = f"offset={params['offset']:.6f}"
            elif transform == 'box_cox' and 'lambda' in params:
                details = f"lambda={params['lambda']:.6f}"
            elif transform == 'yeo_johnson' and 'lambda' in params:
                details = f"lambda={params['lambda']:.6f}"
            else:
                details = "-"

            print(f"{feat:<30} {transform:<15} {details}")
    else:
        print("❌ No transformation config found!")

    print("\n" + "=" * 100)
    print("STAGE 10: FEATURE STANDARDIZATION - What was applied?")
    print("=" * 100)

    # Get standardization config
    std_config = db['feature_standardization_config'].find_one({})
    if std_config:
        selected_hl = std_config.get('selected_halflife', {})
        scaler_stats = std_config.get('scaler_statistics', {})

        # Check if volatility was standardized (BUG!)
        if 'volatility' in selected_hl:
            print(f"\n❌ BUG DETECTED: VOLATILITY WAS STANDARDIZED!")
            print(f"   Half-life: {selected_hl['volatility']}")
            if 'volatility' in scaler_stats:
                stats = scaler_stats['volatility']
                print(f"   EWMA Mean: {stats.get('ewma_mean', 'N/A'):.10f}")
                print(f"   EWMA Std: {(stats.get('ewma_var', 0) ** 0.5):.10f}")
                print(f"   Samples: {stats.get('n_samples', 'N/A')}")

                # If mean is negative, this confirms log transformation
                mean = stats.get('ewma_mean', 0)
                if mean < 0:
                    print(f"   ⚠️  NEGATIVE MEAN confirms volatility was LOG-TRANSFORMED!")
        else:
            print("\n✓ volatility: Correctly excluded from standardization")

        # Show all standardizations
        print(f"\nTotal features standardized: {len(selected_hl)}")
        print("\nStandardization Summary:")
        print(f"{'Feature':<30} {'Half-Life':<12} {'EWMA Mean':<15} {'EWMA Std':<15} {'Samples'}")
        print("-" * 100)

        for feat in sorted(selected_hl.keys()):
            hl = selected_hl[feat]
            if feat in scaler_stats:
                stats = scaler_stats[feat]
                mean = stats.get('ewma_mean', float('nan'))
                var = stats.get('ewma_var', 0.0)
                std = var ** 0.5
                n = stats.get('n_samples', 0)

                # Flag if mean is suspiciously negative
                flag = " ⚠️ NEG" if mean < -0.1 else ""

                print(f"{feat:<30} {hl:<12} {mean:<15.6f} {std:<15.6f} {n}{flag}")
            else:
                print(f"{feat:<30} {hl:<12} {'N/A':<15} {'N/A':<15} {'N/A'}")
    else:
        print("❌ No standardization config found!")

    print("\n" + "=" * 100)
    print("FEATURES OF INTEREST (6, 7, 10, 11, 15)")
    print("=" * 100)

    # Map feature indices to names (need to check actual feature names)
    print("\nTo map feature indices to names, checking split_0_output...")

    # Sample one document to get feature names
    sample = db['split_0_output'].find_one({}, {'features': 1})
    if sample and 'features' in sample:
        features = sample['features']
        print(f"\nTotal features in output: {len(features)}")

        # Get all feature names from the database
        feature_names_doc = db['feature_names'].find_one({})
        if feature_names_doc and 'features' in feature_names_doc:
            feature_names = feature_names_doc['features']
            print(f"\nFeature mapping:")
            for idx in [0, 6, 7, 10, 11, 15]:
                if idx < len(feature_names):
                    name = feature_names[idx]
                    print(f"  Feature {idx}: {name}")

                    # Check if this feature was transformed
                    if name in selected:
                        print(f"    └─> Transformed: {selected[name]}")

                    # Check if this feature was standardized
                    if name in selected_hl:
                        print(f"    └─> Standardized: half-life={selected_hl[name]}")
        else:
            print("Feature names not found in database")

    client.close()

if __name__ == '__main__':
    main()
