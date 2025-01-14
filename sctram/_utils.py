#!/usr/bin/env python3

import logging


class Utils:

    logger: logging.Logger

    def sget(self, dictionary, key, default=None):
        """Retrieves the value associated with a specified key in the given dictionary.

        If the key does not exist, a default value is returned and a warning is logged.

        Parameters:
            dictionary (dict): The dictionary from which to retrieve the value.
            key (any): The key for which to retrieve the value.
            default (any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            any: The value from the dictionary associated with the specified key, or the default value
                if the key is not found.

        Raises:
            Warning: If the default value is used, a warning is issued.
        """
        if key in dictionary:
            return dictionary[key]
        else:
            self.logger.warning(f"Default value {default!r} used for missing key {key!r}.")
            return default
