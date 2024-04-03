# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


class NodeClassRegistry:
    """Keeps track of the node classes that are defined.

    This simple class just stores the classes in a list,
    but allows other registries to "subscribe" to it. This means
    that when a new class is added to the registry it will
    also be registered in the subscribed registries.

    In this way, a user can implement custom behavior to register
    classes. This can be useful for example to build search indexes.
    """

    def __init__(self):
        self.all_classes = []

        self._subscribed = []

    def register(self, node_cls):
        self.all_classes.append(node_cls)
        for sub in self._subscribed:
            sub.register(node_cls)

    def subscribe(self, registry):
        for cls in self.all_classes:
            registry.register(cls)

        self._subscribed.append(registry)

    def unsubscribe(self, registry):
        self._subscribed.remove(registry)


# Define the global main registry
REGISTRY = NodeClassRegistry()
