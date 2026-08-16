"""Test router registration in detail."""
import sys
sys.path.insert(0, '.')

from backend.main import create_app

app = create_app()
print(f"Total routes: {len(app.routes)}")

# Let's examine each route
for i, route in enumerate(app.routes):
    print(f"\nRoute {i}:")
    print(f"  Type: {type(route)}")
    print(f"  Class name: {route.__class__.__name__}")
    
    if hasattr(route, 'path'):
        print(f"  Path: {route.path}")
    elif hasattr(route, 'prefix'):
        print(f"  Prefix: {route.prefix}")
        print(f"  Has {len(route.routes)} sub-routes")
        for j, subroute in enumerate(route.routes):
            print(f"    Subroute {j}: {subroute.path if hasattr(subroute, 'path') else 'no path'}")
    else:
        print(f"  No path or prefix")