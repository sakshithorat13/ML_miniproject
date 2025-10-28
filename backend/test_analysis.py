import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models import ensemble

# Test the ensemble model
result = ensemble.run()

print("=== TESTING ENSEMBLE MODEL ANALYSIS ===")
print("Keys in result:", result.keys())

if "analysis" in result:
    print("\n✅ Analysis section exists!")
    print("Analysis keys:", result["analysis"].keys())
    print("\nGraph interpretation preview:")
    print(result["analysis"]["graph_interpretation"][:200] + "...")
else:
    print("\n❌ Analysis section missing!")

print("\nFull result structure:")
for key in result.keys():
    print(f"- {key}: {type(result[key])}")
