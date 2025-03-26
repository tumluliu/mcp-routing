"""Tests for the geocoding module."""

import pytest
from mcp_routing.geocoding import NominatimGeocoder
from mcp_routing.routing import RoutingEngine, DummyRoutingEngine


def test_nominatim_geocoder_init():
    """Test initialization of the NominatimGeocoder."""
    geocoder = NominatimGeocoder()
    assert geocoder.base_url == "https://nominatim.openstreetmap.org/search"
    assert "User-Agent" in geocoder.headers


@pytest.mark.skip(reason="Makes actual API calls to Nominatim")
def test_nominatim_geocoder_geocode():
    """Test geocoding a location with Nominatim (skip by default to avoid API calls)."""
    geocoder = NominatimGeocoder()

    # Test geocoding a well-known location in Munich
    coords = geocoder.geocode("Marienplatz, Munich")
    assert coords is not None
    assert len(coords) == 2
    assert isinstance(coords[0], float)  # Latitude
    assert isinstance(coords[1], float)  # Longitude

    # Test approximate location
    assert 48.13 < coords[0] < 48.14  # Latitude should be around 48.137
    assert 11.57 < coords[1] < 11.58  # Longitude should be around 11.575


def test_routing_engine_geocode():
    """Test the geocode method of the RoutingEngine class."""
    engine = RoutingEngine()

    # Test hardcoded locations
    coords = engine.geocode("Marienplatz")
    assert coords == (48.1373, 11.5754)

    coords = engine.geocode("Munich Airport")
    assert coords == (48.3537, 11.7750)

    # Test fallback to random generation
    coords = engine.geocode("Some unknown location")
    assert coords is not None
    assert len(coords) == 2
    assert isinstance(coords[0], float)
    assert isinstance(coords[1], float)


def test_dummy_routing_engine_geocode():
    """Test the geocode method of the DummyRoutingEngine class."""
    engine = DummyRoutingEngine()

    # Test inherited method (and Nominatim integration)
    assert hasattr(engine, "geocoder")

    # Test hardcoded locations
    coords = engine.geocode("Marienplatz")
    assert coords is not None
    assert len(coords) == 2

    coords = engine.geocode("Munich Airport")
    assert coords is not None
    assert len(coords) == 2

    # Test fallback to random generation
    coords = engine.geocode("Some unknown location")
    assert coords is not None
    assert len(coords) == 2
