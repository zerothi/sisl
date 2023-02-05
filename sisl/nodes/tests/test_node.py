from sisl.nodes import Node

def test_node_classes_reused():
    def a():
        pass

    x = Node.from_func(a)
    y = Node.from_func(a)

    assert x is y

def test_overwriting_node_inputs():
    """Tests that you can overwrite node inputs such as automatic_recalc by specifying them
    in the function signature"""

    @Node.from_func
    def calc(val: int):
        return val

    @Node.from_func
    def alert_change(val: int, automatic_recalc=True):
        ...

    val = calc(1)
        
    # We feed the node that produces the intermediate value into our alert node 
    my_alert = alert_change(val=val)

    val.get()
    assert my_alert._nupdates == 0
    val.update_inputs(val=2)
    assert my_alert._nupdates == 1