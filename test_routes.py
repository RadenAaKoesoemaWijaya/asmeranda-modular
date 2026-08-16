"""Test script to check all routes in the app."""
import sys
sys.path.insert(0, '.')

from backend.main import create_app

app = create_app()
print(f"Total routes: {len(app.routes)}")

routes = []
for route in app.routes:
    if hasattr(route, 'path'):
        routes.append(route.path)
    elif hasattr(route, 'routes'):
        # This is an included router
        for subroute in route.routes:
            if hasattr(subroute, 'path'):
                routes.append(subroute.path)

print("All actual paths:")
for path in sorted(routes):
    print(f"  {path}")