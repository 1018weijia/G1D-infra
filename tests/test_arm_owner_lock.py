"""Unit tests for exclusive arm ownership locking."""
import os
import tempfile
import unittest

from teleop.utils.arm_owner_lock import ArmOwnerLock, ArmOwnerLockError, acquire_arm_owner_lock


class ArmOwnerLockTests(unittest.TestCase):
    def test_second_acquire_fails_while_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arm.lock")
            first = acquire_arm_owner_lock("collect", path=path)
            try:
                with self.assertRaises(ArmOwnerLockError):
                    acquire_arm_owner_lock("deploy", path=path)
            finally:
                first.release()
            second = acquire_arm_owner_lock("deploy", path=path)
            second.release()

    def test_context_manager_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arm.lock")
            with ArmOwnerLock(path, "collect") as held:
                self.assertTrue(os.path.isfile(path))
                with open(path, encoding="utf-8") as handle:
                    self.assertIn("collect", handle.read())
                self.assertIs(held.owner, "collect")
            again = acquire_arm_owner_lock("deploy", path=path)
            again.release()


if __name__ == "__main__":
    unittest.main()
