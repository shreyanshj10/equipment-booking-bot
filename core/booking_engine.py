"""
Core booking engine used by the AI agent tools.

Each function performs a specific business operation against the database
(list equipment, check availability, create bookings, etc.).  All database
access goes through SQLAlchemy ORM sessions.  Every function returns a
plain string so the agent can relay the result directly to the user.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from db.database import get_session
from db.models import Booking, Equipment, User


# ─── Date / time helpers ─────────────────────────────────────────────────────

_DATE_FMT = "%Y-%m-%d"
_TIME_FMT = "%H:%M"
_DATETIME_FMT = "%Y-%m-%d %H:%M"


def _parse_slot(date: str, start_time: str, end_time: str) -> tuple[datetime, datetime]:
    """
    Convert separate date and time strings into naive datetime objects.

    Raises ValueError if parsing fails — callers should handle this.
    """

    start_dt = datetime.strptime(f"{date} {start_time}", _DATETIME_FMT)
    end_dt = datetime.strptime(f"{date} {end_time}", _DATETIME_FMT)
    return start_dt, end_dt


def _fmt_time(dt: datetime) -> str:
    """Return a time string like '3:00 PM'."""
    return dt.strftime("%I:%M %p").lstrip("0")


def _fmt_date(dt: datetime) -> str:
    """Return a date string like '15 March 2025'."""
    # Use dt.day (an int) instead of the glibc-only "%-d" directive, which
    # raises "Invalid format string" on Windows.
    return f"{dt.day} {dt.strftime('%B %Y')}"


def _fmt_day_month(dt: datetime) -> str:
    """Return a short date string like '5 Jun'."""
    return f"{dt.day} {dt.strftime('%b')}"


# ─── Public API ───────────────────────────────────────────────────────────────


def list_equipment() -> str:
    """
    Return a formatted list of all equipment, availability, and condition.
    """

    with get_session() as session:
        try:
            stmt = select(Equipment).order_by(Equipment.name.asc())
            equipment_list: List[Equipment] = list(session.execute(stmt).scalars())
        except Exception as exc:
            return f"Failed to list equipment due to an internal error: {exc}"

    if not equipment_list:
        return "No equipment found in the system."

    lines = ["📦 Available Equipment:", "─────────────────────"]
    for i, eq in enumerate(equipment_list, start=1):
        lines.append(
            f"{i}. {eq.name} — {eq.available_quantity}/{eq.total_quantity} available ({eq.condition})"
        )
    lines.append("─────────────────────")
    return "\n".join(lines)


def check_availability(
    equipment_name: str, date: str, start_time: str, end_time: str,
    quantity: int = 1,
) -> str:
    """
    Check whether a specific piece of equipment is free for a given time slot.

    Conflict detection uses the standard interval-overlap condition:
      existing.start_time < new_end AND existing.end_time > new_start
    with status == 'active'.  The quantities of all overlapping bookings
    are summed to determine how many units remain available in the slot.
    """

    try:
        start_dt, end_dt = _parse_slot(date, start_time, end_time)
    except ValueError as exc:
        return f"Invalid date or time format: {exc}"

    if end_dt <= start_dt:
        return "End time must be after start time. Please check your time slot."

    if quantity < 1:
        return "Quantity must be at least 1."

    with get_session() as session:
        try:
            eq_stmt = select(Equipment).where(
                func.lower(Equipment.name) == equipment_name.lower()
            )
            equipment = session.execute(eq_stmt).scalar_one_or_none()
            if not equipment:
                return (
                    f"Equipment '{equipment_name}' not found. "
                    "Use list_equipment to see available options."
                )

            if quantity > equipment.total_quantity:
                return (
                    f"❌ Only {equipment.total_quantity} unit(s) of "
                    f"{equipment.name} exist in total."
                )

            conflict_stmt = (
                select(Booking)
                .where(Booking.equipment_id == equipment.id)
                .where(Booking.status == "active")
                .where(Booking.start_time < end_dt)
                .where(Booking.end_time > start_dt)
                .order_by(Booking.start_time.asc())
            )
            conflicts: List[Booking] = list(session.execute(conflict_stmt).scalars())

            total_booked_in_slot = sum(b.quantity for b in conflicts)
            available_in_slot = equipment.total_quantity - total_booked_in_slot

            if available_in_slot <= 0:
                last_conflict = max(conflicts, key=lambda b: b.end_time)
                return (
                    f"❌ All {equipment.total_quantity} unit(s) of {equipment.name} "
                    f"are booked during that time. "
                    f"Next available after {_fmt_time(last_conflict.end_time)}."
                )

            if quantity > available_in_slot:
                return (
                    f"❌ Only {available_in_slot} unit(s) of {equipment.name} "
                    f"available during that time ({total_booked_in_slot} already booked)."
                )

            date_label = _fmt_date(start_dt)
            return (
                f"✅ {equipment.name} is available on {date_label} "
                f"from {_fmt_time(start_dt)}–{_fmt_time(end_dt)} "
                f"({available_in_slot}/{equipment.total_quantity} units free)."
            )
        except Exception as exc:
            return f"Failed to check availability due to an internal error: {exc}"


def _generate_booking_id(session) -> str:
    """
    Generate the next booking ID in the sequence B001, B002, ...

    Queries the current maximum booking_id and increments it instead of
    counting rows, so gaps from cancellations are handled correctly.
    """

    stmt = select(Booking.booking_id)
    all_ids = list(session.execute(stmt).scalars())

    max_num = 0
    for bid in all_ids:
        try:
            num = int(bid[1:])  # strip leading 'B'
            if num > max_num:
                max_num = num
        except (ValueError, TypeError, IndexError):
            continue

    return f"B{max_num + 1:03d}"


def make_booking(
    equipment_name: str,
    date: str,
    start_time: str,
    end_time: str,
    club_name: str,
    booked_by: str,
    quantity: int = 1,
) -> str:
    """
    Create a booking for the specified equipment and time slot.

    Performs a conflict check, uses a DB transaction for all writes, and
    returns a human-readable confirmation or error message.
    """

    try:
        start_dt, end_dt = _parse_slot(date, start_time, end_time)
    except ValueError as exc:
        return f"Invalid date or time format: {exc}"

    if end_dt <= start_dt:
        return "End time must be after start time. Please check your time slot."

    if quantity < 1:
        return "Quantity must be at least 1."

    with get_session() as session:
        try:
            # Resolve equipment by case-insensitive name.
            eq_stmt = select(Equipment).where(
                func.lower(Equipment.name) == equipment_name.lower()
            )
            equipment = session.execute(eq_stmt).scalar_one_or_none()
            if not equipment:
                return (
                    f"Equipment '{equipment_name}' not found. "
                    "Use list_equipment to see available options."
                )

            if quantity > equipment.total_quantity:
                return (
                    f"❌ Only {equipment.total_quantity} unit(s) of "
                    f"{equipment.name} exist in total."
                )

            # Overlap conflict check — sum quantities of all overlapping bookings.
            conflict_stmt = (
                select(Booking)
                .where(Booking.equipment_id == equipment.id)
                .where(Booking.status == "active")
                .where(Booking.start_time < end_dt)
                .where(Booking.end_time > start_dt)
            )
            conflicts: List[Booking] = list(session.execute(conflict_stmt).scalars())
            total_booked_in_slot = sum(b.quantity for b in conflicts)
            available_in_slot = equipment.total_quantity - total_booked_in_slot

            if quantity > available_in_slot:
                return (
                    f"❌ Only {available_in_slot} unit(s) of {equipment.name} "
                    f"available during that time. Please reduce the quantity "
                    f"or choose a different time slot."
                )

            booking_id = _generate_booking_id(session)

            booking = Booking(
                booking_id=booking_id,
                equipment_id=equipment.id,
                club_name=club_name,
                booked_by=booked_by,
                quantity=quantity,
                start_time=start_dt,
                end_time=end_dt,
                status="active",
            )
            session.add(booking)
            equipment.available_quantity -= quantity
            session.commit()

            eq_name = equipment.name  # capture before session closes

        except Exception as exc:
            session.rollback()
            return f"Failed to create booking due to an internal error: {exc}"

    date_label = _fmt_date(start_dt)
    return (
        f"✅ Booking Confirmed!\n"
        f"─────────────────────\n"
        f"Equipment : {eq_name} x{quantity}\n"
        f"Club      : {club_name}\n"
        f"Date      : {date_label}\n"
        f"Time      : {_fmt_time(start_dt)} – {_fmt_time(end_dt)}\n"
        f"Booking ID: {booking_id}\n"
        f"Contact   : {booked_by}\n"
        f"─────────────────────\n"
        f"Save your Booking ID — you will need it to cancel or return."
    )


def get_bookings(club_name: str) -> str:
    """
    Fetch all active bookings for a given club and return a formatted list.

    Uses a case-insensitive partial match on club_name.
    """

    with get_session() as session:
        try:
            stmt = (
                select(Booking)
                .options(selectinload(Booking.equipment))
                .join(Equipment)
                .where(Booking.club_name.ilike(f"%{club_name}%"))
                .where(Booking.status == "active")
                .order_by(Booking.start_time.asc())
            )
            bookings: List[Booking] = list(session.execute(stmt).scalars())

            if not bookings:
                return f"No active bookings found for {club_name}."

            lines = [f"📋 Active Bookings for {club_name}:", "─────────────────────"]
            for b in bookings:
                date_label = _fmt_day_month(b.start_time)
                lines.append(
                    f"{b.equipment.name} x{b.quantity} | {date_label} | "
                    f"{_fmt_time(b.start_time)}–{_fmt_time(b.end_time)} | {b.booked_by}"
                )
            lines.append("─────────────────────")
            return "\n".join(lines)

        except Exception as exc:
            return f"Failed to fetch bookings due to an internal error: {exc}"


def get_booking_history(club_name: str) -> str:
    """
    Fetch past (returned/cancelled) bookings for a given club.

    Uses a case-insensitive partial match on club_name.
    """

    with get_session() as session:
        try:
            stmt = (
                select(Booking)
                .options(selectinload(Booking.equipment))
                .join(Equipment)
                .where(Booking.club_name.ilike(f"%{club_name}%"))
                .where(Booking.status.in_(["returned", "cancelled"]))
                .order_by(Booking.start_time.desc())
            )
            bookings: List[Booking] = list(session.execute(stmt).scalars())

            if not bookings:
                return f"No past bookings found for {club_name}."

            lines = [f"📜 Booking History for {club_name}:", "─────────────────────"]
            for b in bookings:
                date_label = _fmt_day_month(b.start_time)
                status_icon = "🔄" if b.status == "returned" else "❌"
                lines.append(
                    f"{status_icon} {b.equipment.name} x{b.quantity} | {date_label} | "
                    f"{_fmt_time(b.start_time)}–{_fmt_time(b.end_time)} | "
                    f"{b.booked_by} | {b.status}"
                )
            lines.append("─────────────────────")
            return "\n".join(lines)

        except Exception as exc:
            return f"Failed to fetch booking history due to an internal error: {exc}"


def cancel_booking(booking_id: str) -> str:
    """
    Cancel an active booking and release one unit of the associated equipment.
    """

    with get_session() as session:
        try:
            stmt = select(Booking).where(
                Booking.booking_id == booking_id.strip()
            )
            booking = session.execute(stmt).scalar_one_or_none()
            if not booking:
                return f"Booking {booking_id} not found. Please provide the exact Booking ID (e.g. B006)."

            if booking.status != "active":
                return (
                    f"Booking {booking_id} is already {booking.status} "
                    f"and cannot be cancelled."
                )

            eq_stmt = select(Equipment).where(Equipment.id == booking.equipment_id)
            equipment = session.execute(eq_stmt).scalar_one_or_none()

            booking.status = "cancelled"
            if equipment:
                equipment.available_quantity += booking.quantity
            session.commit()

            eq_name = equipment.name if equipment else "the equipment"

        except Exception as exc:
            session.rollback()
            return f"Failed to cancel booking due to an internal error: {exc}"

    return (
        f"✅ Booking {booking_id} has been cancelled. "
        f"{booking.quantity} unit(s) of {eq_name} released."
    )


def return_equipment(booking_id: str) -> str:
    """
    Mark equipment as returned for an active booking and increment availability.
    """

    with get_session() as session:
        try:
            stmt = select(Booking).where(
                Booking.booking_id == booking_id.strip()
            )
            booking = session.execute(stmt).scalar_one_or_none()
            if not booking:
                return f"Booking {booking_id} not found. Please provide the exact Booking ID (e.g. B006)."

            if booking.status != "active":
                return f"Booking {booking_id} is already {booking.status}."

            eq_stmt = select(Equipment).where(Equipment.id == booking.equipment_id)
            equipment = session.execute(eq_stmt).scalar_one_or_none()

            booking.status = "returned"
            if equipment:
                equipment.available_quantity += booking.quantity
            session.commit()

            eq_name = equipment.name if equipment else "the equipment"

        except Exception as exc:
            session.rollback()
            return f"Failed to mark equipment as returned due to an internal error: {exc}"

    return (
        f"✅ Equipment returned successfully. "
        f"Booking {booking_id} marked as returned. "
        f"{booking.quantity} unit(s) of {eq_name} back in the pool."
    )


def get_active_bookings() -> str:
    """
    Retrieve all currently active bookings across all clubs.
    """

    with get_session() as session:
        try:
            stmt = (
                select(Booking)
                .options(selectinload(Booking.equipment))
                .join(Equipment)
                .where(Booking.status == "active")
                .order_by(Booking.start_time.asc())
            )
            bookings: List[Booking] = list(session.execute(stmt).scalars())

            if not bookings:
                return "No active bookings at the moment."

            lines = ["📋 All Active Bookings:", "─────────────────────"]
            for b in bookings:
                date_label = _fmt_day_month(b.start_time)
                lines.append(
                    f"{b.equipment.name} x{b.quantity} | {b.club_name} | "
                    f"{date_label} | {_fmt_time(b.start_time)}–{_fmt_time(b.end_time)} | {b.booked_by}"
                )
            lines.append("─────────────────────")
            return "\n".join(lines)

        except Exception as exc:
            return f"Failed to fetch active bookings due to an internal error: {exc}"


def get_all_booking_history() -> str:
    """
    Retrieve past (returned/cancelled) bookings across all clubs.
    """

    with get_session() as session:
        try:
            stmt = (
                select(Booking)
                .options(selectinload(Booking.equipment))
                .join(Equipment)
                .where(Booking.status.in_(["returned", "cancelled"]))
                .order_by(Booking.start_time.desc())
            )
            bookings: List[Booking] = list(session.execute(stmt).scalars())

            if not bookings:
                return "No past bookings found."

            lines = ["📜 All Booking History:", "─────────────────────"]
            for b in bookings:
                date_label = _fmt_day_month(b.start_time)
                status_icon = "🔄" if b.status == "returned" else "❌"
                lines.append(
                    f"{status_icon} {b.equipment.name} x{b.quantity} | {b.club_name} | "
                    f"{date_label} | {_fmt_time(b.start_time)}–{_fmt_time(b.end_time)} | "
                    f"{b.booked_by} | {b.status}"
                )
            lines.append("─────────────────────")
            return "\n".join(lines)

        except Exception as exc:
            return f"Failed to fetch all booking history due to an internal error: {exc}"


# ─── User management ────────────────────────────────────────────────────────


def lookup_user(username: str) -> dict | None:
    """Look up a user by username (case-insensitive). Returns info dict or None."""

    with get_session() as session:
        stmt = select(User).where(func.lower(User.username) == username.strip().lower())
        user = session.execute(stmt).scalar_one_or_none()
        if not user:
            return None
        return {
            "username": user.username,
            "club_name": user.club_name,
            "role": user.role,
        }


def add_user(username: str, club_name: str) -> str:
    """Add a new regular user assigned to a club."""

    with get_session() as session:
        try:
            stmt = select(User).where(func.lower(User.username) == username.strip().lower())
            existing = session.execute(stmt).scalar_one_or_none()
            if existing:
                return (
                    f"User '{existing.username}' already exists "
                    f"(club: {existing.club_name or 'N/A'}, role: {existing.role})."
                )

            new_user = User(
                username=username.strip(),
                club_name=club_name.strip(),
                role="user",
            )
            session.add(new_user)
            session.commit()
            return f"✅ User '{username.strip()}' added and assigned to {club_name.strip()}."
        except Exception as exc:
            session.rollback()
            return f"Failed to add user: {exc}"


def list_users() -> str:
    """List all users in the system. Admin only."""

    with get_session() as session:
        try:
            stmt = select(User).order_by(User.role, User.username)
            users = session.execute(stmt).scalars().all()
            if not users:
                return "No users found in the system."

            lines = [f"📋 All Users ({len(users)} total)", "─────────────────────"]
            for u in users:
                club_display = u.club_name or "N/A"
                lines.append(f"  {u.username}  |  Club: {club_display}  |  Role: {u.role}")
            lines.append("─────────────────────")
            return "\n".join(lines)
        except Exception as exc:
            return f"Failed to list users: {exc}"


def remove_user(username: str) -> str:
    """Remove a user by username. Cannot remove admin users."""

    with get_session() as session:
        try:
            stmt = select(User).where(func.lower(User.username) == username.strip().lower())
            user = session.execute(stmt).scalar_one_or_none()
            if not user:
                return f"User '{username}' not found."
            if user.role == "admin":
                return f"Cannot remove admin user '{user.username}'."
            session.delete(user)
            session.commit()
            return f"✅ User '{user.username}' has been removed."
        except Exception as exc:
            session.rollback()
            return f"Failed to remove user: {exc}"
