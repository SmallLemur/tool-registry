"""Abstract registration plugin interface.

Registration plugins handle service auto-discovery.
The default plugin (http_push) relies on services POSTing to /api/v1/register.
The RabbitMQ plugin listens on a configurable fanout exchange for manifests.

Both plugins call the same on_register/on_deregister callbacks, keeping the
RegistryManager independent of how services are discovered.

Future plugins could use: Consul, etcd, Kubernetes Service Discovery, etc.
"""

from abc import ABC, abstractmethod
from typing import Callable


class RegistrationPlugin(ABC):
    """Abstract base class for service registration/discovery plugins."""

    @abstractmethod
    async def start(
        self,
        on_register: Callable,
        on_deregister: Callable,
    ) -> None:
        """Start the plugin.

        Args:
            on_register:   async callable(manifest: dict) → called when a service registers
            on_deregister: async callable(service_name: str) → called when a service leaves
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the plugin and release resources."""
        ...
