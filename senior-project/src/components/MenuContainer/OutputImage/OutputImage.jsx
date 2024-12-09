import "./OutputImage.css";

export default function OutputImage(props) {
    return (
        <img src={props.src} onChange={props.onChange} />
    );
}