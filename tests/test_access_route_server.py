"""Verified person routes are private to a visitor and survive a page reload."""

import numpy as np
from fastapi.testclient import TestClient
from test_way import room

from pivot.server.app import Room, create_app


def test_route_endpoint_and_printable_plan_share_the_verified_route():
    app = create_app([Room("room","Room",room(),np.zeros((0,3)),np.zeros((0,3),dtype=np.uint8))])
    first, second = TestClient(app), TestClient(app)
    assert first.get('/api/access-routes').json() == []
    result = first.post('/api/tools/make_way',json={'object_id':2}).json()
    assert result['success']
    routes = first.get('/api/access-routes').json()
    assert routes[0]['object_id'] == 2 and routes[0]['from_door'] == 1
    assert second.get('/api/access-routes').json() == []
    assert 'class="access-route"' in first.get('/instructions').text
    first.get('/')
    assert first.get('/api/access-routes').json() == routes
    first.post('/api/reset')
    assert first.get('/api/access-routes').json() == []
