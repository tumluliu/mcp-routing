"""Tests for the routing module."""

import pytest
from mcp_routing.routing import (
    RoutingEngine,
    OSRMEngine,
    OpenRouteServiceEngine,
    get_routing_engine,
)


def test_routing_engine_base():
    """Test the base RoutingEngine class."""
    engine = RoutingEngine()

    # Test geocoding with a known location
    coords = engine.geocode("Marienplatz")
    assert isinstance(coords, tuple)
    assert len(coords) == 2
    assert isinstance(coords[0], float)  # Latitude
    assert isinstance(coords[1], float)  # Longitude

    # Test that the abstract route method raises NotImplementedError
    with pytest.raises(NotImplementedError):
        engine.route("Marienplatz", "Hauptbahnhof")


def test_get_routing_engine():
    """Test the get_routing_engine factory function."""
    # Test getting OSRM engine
    engine = get_routing_engine("osrm")
    assert isinstance(engine, OSRMEngine)

    # Test getting OpenRouteService engine
    engine = get_routing_engine("openrouteservice")
    assert isinstance(engine, OpenRouteServiceEngine)

    # Test invalid engine name
    with pytest.raises(ValueError):
        get_routing_engine("invalid_engine")


# The following tests are marked as skip by default because they require
# running routing engines. To run them, use:
# pytest -v tests/test_routing.py::test_osrm_engine -k "not skip"


@pytest.mark.skip(reason="Requires running OSRM engine")
def test_osrm_engine():
    """Test the OSRM engine with a simple route."""
    engine = OSRMEngine()

    # Test a simple route from Marienplatz to Hauptbahnhof
    route_data = engine.route("Marienplatz", "Hauptbahnhof")

    # Check that we got a valid response
    assert "distance" in route_data
    assert "duration" in route_data
    assert "geometry" in route_data
    assert "steps" in route_data
    assert "origin" in route_data
    assert "destination" in route_data

    # Check that the route has a reasonable distance
    # Distance from Marienplatz to Hauptbahnhof should be around 1-2 km
    assert 500 < route_data["distance"] < 3000


@pytest.mark.skip(reason="Requires running OpenRouteService engine")
def test_openrouteservice_engine():
    """Test the OpenRouteService engine with a simple route."""
    engine = OpenRouteServiceEngine()

    # Test a simple route from Marienplatz to Hauptbahnhof
    route_data = engine.route("Marienplatz", "Hauptbahnhof")

    # Check that we got a valid response
    assert "distance" in route_data
    assert "duration" in route_data
    assert "geometry" in route_data
    assert "steps" in route_data
    assert "origin" in route_data
    assert "destination" in route_data

    # Check that the route has a reasonable distance
    # Distance from Marienplatz to Hauptbahnhof should be around 1-2 km
    assert 500 < route_data["distance"] < 3000
