export default function FileInput(props) {
    return (
        <input type="file" onChange={props.onChange} className="inline">{props.children}</input>
    );
}