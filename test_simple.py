#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

# Mock pydantic since it's not available
class MockPydantic:
    class BaseModel:
        pass
        
import sys
sys.modules['pydantic'] = MockPydantic()

# Now try to import the module
try:
    from dachi.utils._request import RequestDispatcher, RequestState, RequestStatus
    print("✓ Module imported successfully")
    
    # Test basic functionality
    rd = RequestDispatcher.obj
    print(f"✓ Dispatcher created: {type(rd)}")
    
    # Test a simple function
    def test_func(x):
        return x * 2
        
    req_id = rd.submit_func(test_func, 5)
    print(f"✓ Function submitted: {req_id}")
    
    # Wait a bit and check status
    import time
    time.sleep(0.1)
    status = rd.status(req_id)
    print(f"✓ Status: {status}")
    
    result = rd.result(req_id)
    print(f"✓ Result: {result}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()