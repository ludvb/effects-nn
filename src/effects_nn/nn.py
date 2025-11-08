"""effects_nn.nn"""

import abc
import dataclasses as dc
from typing import Any, Callable, Concatenate, overload

import effects as fx
import effects_state as state


program_store = state.create_state_domain("program_store")


def name[**InputT, OutputT](
    name: str | None = None,
) -> Callable[[Callable[InputT, OutputT]], Callable[InputT, OutputT]]:
    def decorator(fn):
        def wrapper(*args, **kwargs):
            with program_store.scope(name or []):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


@dc.dataclass
class ApplyProgram[OutputT](fx.Effect[OutputT]):
    apply_args: tuple
    apply_kwargs: dict


class _Program[**InputT, OutputT](metaclass=abc.ABCMeta):
    def __init__(self, fn):
        self.fn = fn

    @abc.abstractmethod
    def _call_program(self, *args: InputT.args, **kwargs: InputT.kwargs) -> OutputT: ...

    def __call__(self, *args, _name: str | None = None, **kwargs):
        with program_store.scope(_name or []):
            ns = program_store.get_path()

            @fx.handler
            def default_handler(eff: ApplyProgram):
                if program_store.get_path() != ns:
                    return fx.send(eff, interpret_final=False)
                return self._call_program(*eff.apply_args, **eff.apply_kwargs)

            def _send_apply() -> OutputT:
                return fx.send(ApplyProgram(apply_args=args, apply_kwargs=kwargs))

            return fx.bind(_send_apply, default_handler, *fx.get_stack())()


class _InitProgram[**InputT](_Program[InputT, None]):
    def _call_program(self, *args: InputT.args, **kwargs: InputT.kwargs):
        state = self.fn(*args, **kwargs)
        program_store.update(state=state)

    def __call__(self, *args: InputT.args, **kwargs: InputT.kwargs) -> None:
        return super().__call__(*args, **kwargs)


class _PureProgram[**InputT, OutputT](_Program[InputT, OutputT]):
    def _call_program(self, *args: InputT.args, **kwargs: InputT.kwargs) -> OutputT:
        state = program_store.get("state")
        return self.fn(state, *args, **kwargs)

    def __call__(self, *args: InputT.args, **kwargs: InputT.kwargs) -> OutputT:
        return super().__call__(*args, **kwargs)


class _ImpureProgram[**InputT, OutputT](_Program[InputT, OutputT]):
    def _call_program(self, *args: InputT.args, **kwargs: InputT.kwargs) -> OutputT:
        state = program_store.get("state")
        state, out = self.fn(state, *args, **kwargs)
        program_store.update(state=state)
        return out

    def __call__(self, *args: InputT.args, **kwargs: InputT.kwargs) -> OutputT:
        return super().__call__(*args, **kwargs)


def init_program[**InputT](fn: Callable[InputT, Any]) -> _InitProgram[InputT]:
    return _InitProgram(fn)


def pure_program[**InputT, OutputT](
    fn: Callable[Concatenate[Any, InputT], OutputT],
) -> _PureProgram[InputT, OutputT]:
    return _PureProgram(fn)


def impure_program[StateT, **InputT, OutputT](
    fn: Callable[Concatenate[StateT, InputT], tuple[StateT, OutputT]],
) -> _ImpureProgram[InputT, OutputT]:
    return _ImpureProgram(fn)


@overload
def unlift_program[**InputT](
    program: _InitProgram[InputT],
) -> Callable[InputT, dict[Any, Any]]: ...


@overload
def unlift_program[**InputT, OutputT](
    program: _PureProgram[InputT, OutputT],
) -> Callable[Concatenate[Any, InputT], OutputT]: ...


@overload
def unlift_program[**InputT, OutputT](
    program: _ImpureProgram[InputT, OutputT],
) -> Callable[Concatenate[Any, InputT], tuple[Any, OutputT]]: ...


def unlift_program[**InputT, OutputT](
    program: (
        _InitProgram[InputT]
        | _PureProgram[InputT, OutputT]
        | _ImpureProgram[InputT, OutputT]
    ),
):
    match program:
        case _InitProgram():
            fn_pure = program_store.unlift(program)

            def _unlifted(*args: InputT.args, **kwargs: InputT.kwargs):
                out, _ = fn_pure({}, *args, **kwargs)
                return out

            return _unlifted

        case _PureProgram():
            return program_store.unlift(program, state_is_mutated=False)

        case _ImpureProgram():
            return program_store.unlift(program)

        case _:
            raise ValueError(f"Unknown program type: {type(program)}")
