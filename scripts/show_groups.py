import joblib, pathlib, collections
gfile = pathlib.Path("data")/"groups.pkl"
if gfile.exists():
    g = joblib.load(gfile)
    print("Group counts by Study:", dict(collections.Counter(g)))
else:
    print("No groups.pkl found")
